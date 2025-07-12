# --- START OF FILE ai_processor.py (v2.1 - 恢復原始模型，僅優化 Prompt) ---

import base64
import os
from google import genai
from google.genai import types
from PIL import Image
import json
import re
import traceback
import pymysql
from pymysql.err import MySQLError as Error
from typing import List, Dict, Any, Tuple
import time
import difflib
from collections import defaultdict

# --- 輔助函式 (无变动) ---
def get_db_connection(db_config: dict):
    try:
        return pymysql.connect(**db_config)
    except Error as e:
        print(f"  [DB] 資料庫連線失敗: {e}")
        return None

def extract_keywords_from_ocr_data(ocr_data: List[Dict[str, Any]]) -> List[str]:
    keywords = set()
    pattern = re.compile(r'([a-zA-Z]{3,})|([\u4e00-\u9fa5]{2,})')
    for item in ocr_data:
        line = item['text']
        processed_line = line.replace('.', '')
        matches = pattern.findall(processed_line)
        for en_word, zh_word in matches:
            if en_word and en_word.upper() not in ['TABLET', 'CAPSULE', 'MG', 'CC', 'ORAL', 'FOR', 'USE']:
                keywords.add(en_word)
            if zh_word and zh_word not in ['藥品', '用法', '每日', '飯前', '飯後', '睡前', '服用', '單位', '總數', '藥袋', '診所', '醫師', '姓名', '劑量', '天數']:
                keywords.add(zh_word)
    return list(keywords)

def get_filtered_drug_database(conn, keywords: List[str]) -> List[Dict[str, Any]]:
    if not keywords: return []
    drug_list = []
    try:
        with conn.cursor() as cursor:
            where_clauses, params = [], []
            for keyword in keywords:
                kw_no_dot = keyword.replace('.', '')
                where_clauses.extend([
                    "LOWER(REPLACE(drug_name_en, '.', '')) LIKE LOWER(%s)",
                    "drug_name_zh LIKE %s"
                ])
                params.extend([f"%{kw_no_dot}%", f"%{keyword}%"])
            query = f"""
                SELECT drug_id, drug_name_zh, drug_name_en, main_use, side_effects
                FROM drug_info
                WHERE {' OR '.join(where_clauses)}
            """
            cursor.execute(query, tuple(params))
            drug_list = cursor.fetchall()
    except Error as e:
        print(f"  [DB] 篩選藥物參考清單失敗: {e}", flush=True)
    return drug_list

# --- 後處理函式 (維持原樣) ---
def simple_post_process(analysis_result: Dict, conn) -> Dict:
    print("  [後處理] 開始進行簡單驗證與格式化...", flush=True)
    if not conn or not isinstance(analysis_result, dict) or not analysis_result.get("medications"):
        return analysis_result
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT frequency_code, frequency_name FROM frequency_code")
            freq_map = {row['frequency_code']: row['frequency_name'] for row in cursor.fetchall()}

            for med in analysis_result.get("medications", []):
                code = med.get('frequency_count_code')
                if code and not med.get('frequency_text'):
                    med['frequency_text'] = freq_map.get(code, code)
                
                if med.get('matched_drug_id'):
                     cursor.execute("SELECT main_use, side_effects FROM drug_info WHERE drug_id = %s", (med['matched_drug_id'],))
                     db_info = cursor.fetchone()
                     if db_info:
                         if not med.get('main_use'):
                             med['main_use'] = db_info.get('main_use')
                         if not med.get('side_effects'):
                             med['side_effects'] = db_info.get('side_effects')
    except Error as e:
        print(f"  [後處理] 執行簡單驗證時出錯: {e}", flush=True)
        
    print("  [後處理] 簡單驗證與格式化完成。", flush=True)
    return analysis_result

def final_validation_and_correction(analysis_result: Dict, ocr_data: List[Dict[str, Any]]) -> Dict:
    """
    對 AI 分析結果進行最後的、基於規則的確定性校驗與校正。
    特別處理將「天數」誤判為「劑量」的情況。
    """
    print("  [最終校驗] 開始執行確定性校驗與校正...", flush=True)
    if not isinstance(analysis_result, dict) or not analysis_result.get("medications"):
        return analysis_result

    # 為了方便查找，將 OCR 資料按 Y 座標（行）和 X 座標（列）排序
    sorted_ocr = sorted(ocr_data, key=lambda item: (item['box'][1], item['box'][0]))

    # 1. 嘗試從 OCR 全文中提取一個全局的天數
    global_days_supply = None
    for item in sorted_ocr:
        # 尋找像 "* 3" 或 "共3日份" 這樣的模式
        match = re.search(r'[*xX]\s*(\d+)|共(\d+)日', item['text'])
        if match:
            # group(1)對應 * 3, group(2)對應共3日
            days_str = match.group(1) or match.group(2)
            if days_str and days_str.isdigit():
                global_days_supply = int(days_str)
                print(f"  [最終校驗] 從文本 '{item['text']}' 中提取到全局天數: {global_days_supply}", flush=True)
                # 通常藥單只會有一個天數，找到第一個就跳出
                break
    
    # 如果找到了全局天數，就將其設定到頂層
    if global_days_supply and not analysis_result.get('days_supply'):
        analysis_result['days_supply'] = str(global_days_supply)

    # 2. 遍歷每一種藥物，校驗其劑量
    for med in analysis_result.get("medications", []):
        try:
            # 獲取劑量字串並嘗試轉為數字
            dose_str = str(med.get('dose_quantity', '')).strip()
            # 從 "3 錠" 中只取出數字部分
            dose_value_match = re.match(r'(\d+\.?\d*)', dose_str)
            if not dose_value_match:
                continue

            dose_value = float(dose_value_match.group(1))

            # 【校驗規則】如果劑量是一個大於2的整數，且全局天數與之相等，這就是一個高度可疑的信號
            if dose_value > 2 and dose_value == global_days_supply:
                print(f"  [最終校驗] 發現可疑劑量！藥物: '{med.get('drug_name_zh')}', 劑量: {dose_value}, 全局天數: {global_days_supply}", flush=True)
                
                # 在這種情況下，我們自信地將其校正
                # 尋找原始劑量單位，如果沒有，就預設為 '錠'
                original_unit_match = re.search(r'\d+\.?\d*\s*(.*)', dose_str)
                unit = original_unit_match.group(1).strip() if original_unit_match and original_unit_match.group(1) else '錠'
                
                corrected_dose_str = f"1 {unit}"
                med['dose_quantity'] = corrected_dose_str
                print(f"  [最終校驗] 已將劑量校正為: '{corrected_dose_str}'", flush=True)

        except (ValueError, TypeError) as e:
            # 如果 dose_quantity 不是數字（例如 "半顆"），則跳過校驗
            print(f"  [最終校驗] 跳過非數字劑量的校驗: {med.get('dose_quantity')}, error: {e}", flush=True)
            continue
            
    print("  [最終校驗] 確定性校驗與校正完成。", flush=True)
    return analysis_result

# --- 核心分析函式 ---
def run_analysis(ocr_data_with_boxes: List[Dict[str, Any]], db_config: dict, api_key: str) -> Tuple[Dict | None, Dict | None]:
    start_time = time.time()
    print("[AI] 開始 'AI驅動解析' 模式任務...", flush=True)
    if not ocr_data_with_boxes: return None, None
    conn = get_db_connection(db_config)
    if not conn: return None, None
    try:
        with conn.cursor() as cursor:
            keywords = extract_keywords_from_ocr_data(ocr_data_with_boxes)
            candidates = get_filtered_drug_database(conn, keywords)
            cursor.execute("SELECT frequency_code, frequency_name FROM frequency_code")
            freq_codes = cursor.fetchall()
        
        ocr_data_str = json.dumps(ocr_data_with_boxes, ensure_ascii=False, indent=2, default=str)
        drug_ref_str = json.dumps(candidates, ensure_ascii=False, indent=2, default=str)
        freq_ref_str = json.dumps(freq_codes, ensure_ascii=False, indent=2, default=str)

        # --- AI 模型框架 (維持您指定的版本) ---
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-flash" # 【已恢復】嚴格使用您指定的模型

        # --- Prompt 優化版 ---
        prompt_lines = [
            "# **角色与核心任务**",
            "你是一位顶级的、能理解空间布局和上下文的药单解析专家。你的任务是根据我提供的、带有精确座标和来源文件索引 (`source_index`) 的OCR文本块列表，以及候选药物和频率代码参考，为**每一份独立的药袋（由`source_index`区分）**生成一份结构化的JSON报告。你必须极其严谨，**尽最大努力**从文本中推断并补全所有关键信息。",
            
            "---",
            "# **输入资料**",
            "1.  **OCR Data**: 一个JSON列表，每个条目包含`text`, `box`座标, `source_index`来源文件索引。",
            "2.  **Drug Candidates**: 一个JSON列表，包含可能出现在药单上的药物信息（`drug_id`, `drug_name_zh`, `drug_name_en`）。",
            "3.  **Frequency Codes**: 一个JSON列表，用于将文本描述转换为标准代码。",
            "---",
            "# **核心指令与推理规则 (必须严格遵守)**",
            "1.  **分药袋处理**: 你的最终输出必须是一个JSON列表 `[]`，列表中的每个物件 `{}` 代表一个独立的药袋（即一个 `source_index`）。对于每一个 `source_index`，执行以下操作：",
            "    a.  **提取全局信息**: 在当前 `source_index` 的所有文本块中，提取`clinic_name`, `doctor_name`, `visit_date`。",
            "    b.  **识别所有药物**: 在当前 `source_index` 的文本块中，找出所有出现的藥物，存入`medications`中，並為每種藥物生成一個獨立的條目。",
            "2.  **补全药项细节 (最关键的推理部分)**:",
            "    *   **定位锚点**: 对于每种药物，在其所属的 `source_index` 内，找到它的名称文本块作為分析的“錨點”。",
            "    *   **關聯上下文**: 仔細分析錨點**周圍**的所有文本塊，即使它們不直接相鄰。",
            "    *   **【用法推理規則】**: 如果只看到明確的次數（如“每日二次”），就填充 `frequency_text` 和 `frequency_count_code` (`BID`)。如果只看到用餐時間（如“飯後”），**你必須進行合理推斷**：通常意味著一日三次，所以請將 `frequency_count_code` 設為 `TID`，`frequency_timing_code` 設為 `PC`。",
            "    *   **【劑量推理規則】**: 尋找“每次 N 顆/錠/包”或“N#”這樣的描述。`dose_quantity` 字段**必須**包含數字和單位（例如 '0.5 顆'）。如果只看到數字（如 '半'），你必須在周圍尋找單位（如 '顆', '錠', 'Tab'）並**主動組合**它們。**'半' 應該被理解為 '0.5'。**",
            "3.  **格式化輸出**: 整合所有信息。**如果經過所有推理，某個信息依然找不到任何直接或間接的證據，才可以將對應字段設為 `null`。**",
            
            "---",
            "# **範例學習 (包含困難案例)**",
            "## **案例1 (信息不全)**: OCR Data 中有 `{'text': '臟得樂', 'source_index': 1}`, `{'text': '飯後', 'source_index': 1}`, `{'text': '半顆', 'source_index': 1}`",
            "## **你的推理過程**: 看到了'飯後'，但沒看到次數 -> 推斷為一日三次 (TID)。看到了'半顆' -> 組合成 '0.5 顆'。",
            "## **你的輸出**: `{'drug_name_zh': '臟得樂錠', 'dose_quantity': '0.5 顆', 'frequency_count_code': 'TID', 'frequency_timing_code': 'PC'}`",
            
            "## **案例2 (信息完整)**: OCR Data 中有 `{'text': '每日 一次', 'source_index': 0}`, `{'text': '飯後', 'source_index': 0}`, `{'text': '1顆', 'source_index': 0}`, `{'text': 'Entresto', 'source_index': 0}`",
            "## **你的輸出**: `{'drug_name_zh': '健安心100毫克膜衣錠', 'dose_quantity': '1 顆', 'frequency_text': '每日一次', 'frequency_count_code': 'QD', 'frequency_timing_code': 'PC', '主要用途': '治感冒', '副作用: 嗜睡'}`",

            "---",
            "確保你接收到的資訊為藥單或處方籤，相關資訊例子如下:診所名稱、醫生名稱、藥物名稱，如果你接收到的OCR Data缺少任一資訊，就不用做分析了",
            "# **开始分析**: 请处理以下真实数据，并严格按照要求的JSON列表格式返回结果。",
            "## 1. OCR Data:", "```json", ocr_data_str, "```",
            "## 2. Drug Candidates:", "```json", drug_ref_str, "```",
            "## 3. Frequency Codes:", "```json", freq_ref_str, "```",
            "## **你的最终输出 (JSON列表格式)**:",
            "source_index:",
            "clinic_name:",
            "doctor_name:",
            "medication:",
            "visit_date:"
            ]
        
        full_prompt = "\n".join(prompt_lines)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])]
        
        # --- AI 模型框架 (維持您指定的配置) ---
        config = types.GenerateContentConfig(temperature=0, thinking_config=types.ThinkingConfig(thinkingBudget = 0), response_mime_type="application/json")
        chunks = client.models.generate_content_stream(model=model_name, contents=contents, config=config)
        response_text = "".join([chunk.text for chunk in chunks if hasattr(chunk, "text") and chunk.text])
        
        print("  [AI Raw Output] AI返回的原始JSON:", flush=True)
        print(response_text)
        
        ai_initial_draft = json.loads(response_text)
        
        # 【方案A：合併結果】
        print("  [方案A] 開始執行多藥袋結果合併...", flush=True)
        final_result = {"medications": []}
        if isinstance(ai_initial_draft, list) and ai_initial_draft:
            first_bag = ai_initial_draft[0]
            final_result['clinic_name'] = first_bag.get('clinic_name')
            final_result['doctor_name'] = first_bag.get('doctor_name')
            final_result['visit_date'] = first_bag.get('visit_date')
            
            for bag in ai_initial_draft:
                if bag.get("medications"):
                    final_result["medications"].extend(bag["medications"])

            for bag in ai_initial_draft:
                if bag.get("days_supply"):
                    final_result['days_supply'] = bag.get("days_supply")
                    break
        else:
            final_result = ai_initial_draft

        # 對合併後的單一結果進行後處理
        with conn:
            processed_result = simple_post_process(final_result, conn)
        
        # 【★★ 核心修改 ★★】調用新的確定性校驗函式
        analysis_result = final_validation_and_correction(processed_result, ocr_data_with_boxes)

        end_time = time.time()
        usage_info = {"model": model_name, "execution_time": end_time - start_time}
        print("-" * 50, flush=True)
        print(f"[AI USAGE] Model: {model_name}", flush=True)
        print(f"[AI USAGE] Execution Time: {usage_info['execution_time']:.4f} seconds", flush=True)
        print("-" * 50, flush=True)
        return analysis_result, usage_info
    except Exception as e:
        print(f"  [AI] 分析處理錯誤: {e}", flush=True); traceback.print_exc()
        return None, None

# --- END OF FILE ai_processor.py (v2.1 - 恢復原始模型，僅優化 Prompt) ---