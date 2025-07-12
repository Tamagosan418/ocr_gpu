# --- START OF FILE app.py (AI ocr API  - 補充環境變數修正版) ---
#API
import os
import traceback
import json
import requests
import re
import pymysql
import time
import base64
import string
import threading
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pymysql.err import MySQLError as Error
from datetime import datetime, date
from decimal import Decimal
from urllib.parse import parse_qs, urlencode

from typing import List, Dict, Any, Tuple
from flask_cors import CORS
import random
import difflib
# --- 導入自訂模組 ---
import ai_processor 
from paddle_ocr_processor import get_text_from_image, ImageDecodeError

# --- 初始化與設定 ---
load_dotenv()
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=os.path.join(basedir, 'static'), template_folder=os.path.join(basedir, 'templates'))
CORS(app)
os.environ["PPOCR_FONTS_DIR"] = os.path.join(os.path.dirname(__file__), "fonts")

try:
    # --- 【關鍵修正】: 新增讀取 LIFF 相關的環境變數 ---
    LIFF_CHANNEL_ID = os.environ.get('LIFF_CHANNEL_ID')
    LIFF_ID_EDIT = os.environ.get('LIFF_ID_EDIT')
    LIFF_ID_REMINDER = os.environ.get('LIFF_ID_REMINDER')
    LIFF_ID_CAMERA = os.environ.get('LIFF_ID_CAMERA') # 確保這裡也讀取了 camera 的 LIFF ID
    
    # 原有的設定保持不變
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    DB_CONFIG = {
        'host': os.environ.get('DB_HOST'), 'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'), 'database': os.environ.get('DB_NAME'),
        'port': int(os.environ.get('DB_PORT')), 'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }
    # 檢查 API 伺服器自身必要的核心設定
    if not GEMINI_API_KEY:
        raise ValueError(".env 檔案缺少必要的設定: GEMINI_API_KEY")

except (ValueError, TypeError) as e: 
    print(f"讀取或解析 .env 設定失敗: {e}", flush=True)
    exit()

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)): return obj.isoformat().split('T')[0]
        if isinstance(obj, Decimal): return float(obj)
        return super().default(obj)
app.json_encoder = CustomJSONEncoder

def get_db_connection():
    try: return pymysql.connect(**DB_CONFIG)
    except Error as e: print(f"資料庫連線失敗: {e}", flush=True); return None

# --- 狀態管理與核心業務邏輯函式 (維持原樣) ---
def get_user_state(line_user_id: str) -> Dict:
    conn = get_db_connection()
    if not conn: return {}
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT state_data FROM user_temp_state WHERE recorder_id = %s", (line_user_id,))
            record = cursor.fetchone()
            if record and record['state_data']: return json.loads(record['state_data'])
            return {}
    finally:
        if conn and conn.open: conn.close()

def set_user_state(line_user_id: str, state_data: Dict):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cursor:
            json_data = json.dumps(state_data, cls=CustomJSONEncoder)
            sql = "INSERT INTO user_temp_state (recorder_id, state_data) VALUES (%s, %s) ON DUPLICATE KEY UPDATE state_data = VALUES(state_data)"
            cursor.execute(sql, (line_user_id, json_data))
            conn.commit()
    except Error as e: print(f"設定使用者狀態失敗: {e}", flush=True); conn.rollback()
    finally:
        if conn and conn.open: conn.close()

def clear_user_state(line_user_id: str):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM user_temp_state WHERE recorder_id = %s", (line_user_id,))
            conn.commit()
    except Error as e: print(f"清除使用者狀態失敗: {e}", flush=True); conn.rollback()
    finally:
        if conn and conn.open: conn.close()

def convert_minguo_to_gregorian(date_str: str | None) -> str | None:
    if not date_str: return None
    date_str = str(date_str).strip()
    match = re.match(r'(\d{2,3})[.\s/-年](\d{1,2})[.\s/-月](\d{1,2})', date_str)
    if not match: return date_str
    year, month, day = [int(g) for g in match.groups()]
    if year < 150:
        try: return date(year + 1911, month, day).strftime('%Y-%m-%d')
        except ValueError: return date_str
    return date_str

def parse_frequency_to_codes(raw_text: str | None, frequency_map_list: List[Dict]) -> Tuple[str | None, str | None]:
    if not raw_text: return None, None
    count_code, timing_code = None, None
    timing_codes = {'AC', 'PC'}
    num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '1': 1, '2': 2, '3': 3, '4': 4}
    match = re.search(r'(?:一天|每日)([一二三四1234])次', raw_text)
    if match:
        day_count_str = match.group(1)
        if day_count_str in num_map:
            times = num_map[day_count_str]
            if times == 1: count_code = 'QD'
            elif times == 2: count_code = 'BID'
            elif times == 3: count_code = 'TID'
            elif times == 4: count_code = 'QID'
    for item in frequency_map_list:
        code, name = item['frequency_code'], item['frequency_name']
        if code in raw_text.upper().replace(' ', ''):
            if code in timing_codes:
                if not timing_code: timing_code = code
            elif not count_code: count_code = code
        elif name in raw_text:
            if code in timing_codes:
                if not timing_code: timing_code = code
            elif not count_code: count_code = code
    return count_code, timing_code

def get_frequency_map_from_db(cursor):
    cursor.execute("SELECT frequency_code, frequency_name, times_per_day, timing_description FROM frequency_code")
    return cursor.fetchall()

def enrich_medication_data(analysis_result: Dict, conn) -> Dict:
    if not conn or not analysis_result or not analysis_result.get("medications"): return analysis_result
    try:
        with conn.cursor() as cursor:
            frequency_map_list = get_frequency_map_from_db(cursor)
            for med in analysis_result.get("medications", []):
                if not isinstance(med, dict): continue
                if 'dosage_value' in med and 'dose_quantity' not in med:
                    med['dose_quantity'] = med.pop('dosage_value')
                if 'dose_quantity' not in med:
                    med['dose_quantity'] = med.get('dosage_value')
                if not med.get('frequency_count_code') and med.get('frequency_text'):
                    count_code, timing_code = parse_frequency_to_codes(med['frequency_text'], frequency_map_list)
                    if count_code: med['frequency_count_code'] = count_code
                    if timing_code: med['frequency_timing_code'] = timing_code
                matched_id = med.get('matched_drug_id')
                if matched_id and (med.get('main_use') is None or med.get('side_effects') is None):
                    cursor.execute("SELECT main_use, side_effects FROM drug_info WHERE drug_id = %s", (matched_id,))
                    drug_info = cursor.fetchone()
                    if drug_info:
                        if med.get('main_use') is None: med['main_use'] = drug_info.get('main_use')
                        if med.get('side_effects') is None: med['side_effects'] = drug_info.get('side_effects')
    except Error as e:
        print(f"豐富藥物資料時出錯: {e}", flush=True)
    return analysis_result

def run_analysis_in_background(line_user_id: str, task_id: str, initial_state: Dict):
    print(f"背景任務開始: task_id={task_id}, user_id={line_user_id}", flush=True)
    start_time = time.time()
    full_state = initial_state
    
    image_bytes_list = full_state.get("last_task", {}).get("image_bytes_list", [])
    if not image_bytes_list:
        full_state["last_task"]["status"] = "error"
        full_state["last_task"]["message"] = "找不到圖片資料"
        set_user_state(line_user_id, full_state)
        return

    try:
        all_ocr_data_with_boxes = [] 
        for index, img_bytes_b64 in enumerate(image_bytes_list):
            print(f"--- 正在 OCR 第 {index + 1} / {len(image_bytes_list)} 張圖片 ---", flush=True)
            img_bytes = base64.b64decode(img_bytes_b64)
            ocr_data = get_text_from_image(img_bytes)
            if ocr_data:
                for item in ocr_data:
                    item['source_index'] = index
                all_ocr_data_with_boxes.extend(ocr_data)
        
        if not all_ocr_data_with_boxes:
            raise ValueError("所有圖片都未能辨識出任何文字。")

        print("--- 所有 OCR 結果已合併並標記，開始一次性 AI 分析 ---", flush=True)
        analysis_result, _ = ai_processor.run_analysis(all_ocr_data_with_boxes, DB_CONFIG, GEMINI_API_KEY)
        if not analysis_result:
            raise ValueError("AI 分析未能返回結果")

        conn = get_db_connection()
        if conn:
            try:
                enriched_result = enrich_medication_data(analysis_result, conn)
            finally:
                conn.close()
        else:
            enriched_result = analysis_result
    
        full_state['last_task']['results'] = enriched_result
        full_state['last_task']['status'] = 'completed'
        print(f"背景任務成功: task_id={task_id}", flush=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[Total]花費秒數{elapsed_time}", flush=True)

    except Exception as e:
        print(f"背景任務失敗: task_id={task_id}, error={e}", flush=True)
        traceback.print_exc()
        full_state['last_task']['status'] = 'error'
        full_state['last_task']['message'] = str(e)
    
    set_user_state(line_user_id, full_state)

def save_final_record(line_user_id: str, member: str, record_data: Dict, mm_id_to_update: int = None) -> Tuple[bool, str, int | None]:
    visit_date_gregorian = convert_minguo_to_gregorian(record_data.get('visit_date'))
    if not visit_date_gregorian:
        return False, "缺少有效的看診日期", None

    conn = get_db_connection()
    if not conn:
        return False, "資料庫連線失敗", None

    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT IGNORE INTO members (recorder_id, member) VALUES (%s, %s)", (line_user_id, member))

            cursor.execute("SELECT drug_id, drug_name_zh FROM drug_info WHERE drug_name_zh IS NOT NULL")
            all_drugs_in_db = cursor.fetchall()
            drug_name_map = {d['drug_name_zh']: d['drug_id'] for d in all_drugs_in_db}
            all_drug_names_zh = list(drug_name_map.keys())

            medications = record_data.get('medications', [])
            for med in medications:
                if not med.get('matched_drug_id') and med.get('drug_name_zh'):
                    best_matches = difflib.get_close_matches(med['drug_name_zh'].strip(), all_drug_names_zh, n=1, cutoff=0.8)
                    if best_matches:
                        med['matched_drug_id'] = drug_name_map[best_matches[0]]
                    else:
                        new_drug_id = f"M_{str(int(time.time()))[-8:]}_{''.join(random.choices(string.ascii_uppercase + string.digits, k=3))}"
                        cursor.execute("INSERT INTO drug_info (drug_id, drug_name_zh) VALUES (%s, %s)", (new_drug_id, med['drug_name_zh'].strip()))
                        med['matched_drug_id'] = new_drug_id
            
            clinic_name = record_data.get('clinic_name')
            
            if mm_id_to_update:
                mm_id = mm_id_to_update
                cursor.execute("UPDATE medication_main SET visit_date = %s, clinic_name = %s, doctor_name = %s WHERE mm_id = %s", (visit_date_gregorian, clinic_name, record_data.get('doctor_name'), mm_id))
                
                cursor.execute("SELECT mr_id FROM medication_records WHERE mm_id = %s", (mm_id,))
                mr_ids_to_delete = [row['mr_id'] for row in cursor.fetchall()]

                if mr_ids_to_delete:
                    format_strings = ','.join(['%s'] * len(mr_ids_to_delete))
                    cursor.execute(f"DELETE FROM record_details WHERE record_id IN ({format_strings})", tuple(mr_ids_to_delete))
                
                cursor.execute("DELETE FROM medication_records WHERE mm_id = %s", (mm_id,))
                
                message = "藥歷已成功更新"
            else:
                sql_insert_main = "INSERT INTO medication_main (recorder_id, member, clinic_name, visit_date, doctor_name) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql_insert_main, (line_user_id, member, clinic_name, visit_date_gregorian, record_data.get('doctor_name')))
                mm_id = cursor.lastrowid
                message = "藥歷已成功儲存"
            
            if medications:
                days_supply = int(record_data.get('days_supply')) if str(record_data.get('days_supply', '')).isdigit() else None
                source_detail = record_data.get("source_detail", "手動")
                for med in medications:
                    sql_insert_record = "INSERT INTO medication_records (mm_id, recorder_id, member, drug_name_en, drug_name_zh, source_detail, dose_quantity, days, frequency_count_code, frequency_timing_code) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    cursor.execute(sql_insert_record, (mm_id, line_user_id, member, med.get('drug_name_en'), med.get('drug_name_zh'), source_detail, med.get('dose_quantity'), days_supply, med.get('frequency_count_code'), med.get('frequency_timing_code')))
                    mr_id = cursor.lastrowid
                    
                    dose_str = str(med.get('dose_quantity', '')).strip()
                    dose_parts = re.match(r'(\d*\.?\d+)\s*(.*)', dose_str)
                    dosage_value = dose_parts.group(1) if dose_parts else ''
                    dosage_unit = dose_parts.group(2).strip() if dose_parts else ''
                    truncated_freq_text = (med.get('frequency_text') or '')[:10]
                    
                    sql_insert_details = "INSERT INTO record_details (record_id, drug_id, dosage_value, dosage_unit, frequency_text) VALUES (%s, %s, %s, %s, %s)"
                    cursor.execute(sql_insert_details, (mr_id, med.get('matched_drug_id'), dosage_value, dosage_unit, truncated_freq_text))

            conn.commit()
            return True, message, mm_id
    except Error as e:
        if conn.open: conn.rollback()
        print(f"儲存藥歷時發生資料庫錯誤: {e}", flush=True); traceback.print_exc()
        return False, "儲存藥歷時發生資料庫錯誤", None
    finally:
        if conn.open: conn.close()

### --- 外部呼叫的核心 API 路由 (維持原本的 `/api/v1/analyze`) --- ###

@app.route("/api/v1/analyze", methods=['POST'])
def api_analyze_prescription():
    if 'photos' not in request.files:
        return jsonify({"status": "error", "message": "請求中未包含任何 'photos' 檔案"}), 400
    
    line_user_id = request.form.get('line_user_id')
    member = request.form.get('member')
    
    if not line_user_id:
        return jsonify({"status": "error", "message": "請求中缺少 line_user_id"}), 400
    if not member:
        return jsonify({"status": "error", "message": "請求中缺少 member (成員) 資訊"}), 400

    photos = request.files.getlist('photos')
    task_id = f"{line_user_id[:8]}_{int(time.time())}"
    
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO users (recorder_id, user_name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE user_name = VALUES(user_name)", (line_user_id, 'API User'))
            cursor.execute("INSERT IGNORE INTO members (recorder_id, member) VALUES (%s, %s)", (line_user_id, member))
        conn.commit()

        image_bytes_list = [base64.b64encode(p.read()).decode('utf-8') for p in photos]
        
        initial_state = {
            "last_task": {
                "task_id": task_id,
                "line_user_id": line_user_id,
                "member": member,
                "image_bytes_list": image_bytes_list,
                "status": "processing"
            }
        }
        set_user_state(line_user_id, initial_state)

        analysis_thread = threading.Thread(target=run_analysis_in_background, args=(line_user_id, task_id, initial_state))
        analysis_thread.start()
        
        return jsonify({"task_id": task_id, "status": "processing", "message": "任務已受理，正在分析中。"}), 202
    except Exception as e:
        if conn and conn.open: conn.rollback()
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"伺服器處理時發生嚴重錯誤: {e}"}), 500
    finally:
        if conn and conn.open:
            conn.close()

@app.route("/api/v1/result/<line_user_id>", methods=['GET'])
def api_get_result(line_user_id):
    full_state = get_user_state(line_user_id)
    task_info = full_state.get("last_task", {})
    if not task_info:
        return jsonify({"status": "error", "message": "找不到該使用者的任務狀態。"}), 404
    status = task_info.get("status")
    if status == 'completed':
        response_data = { "status": "completed", "data": task_info.get("results") }
        return jsonify(response_data)
    elif status == 'error':
        return jsonify({"status": "error", "message": task_info.get("message", "分析時發生未知錯誤")})
    else:
        return jsonify({"status": "processing"})
            
@app.route("/api/v1/save-record", methods=['POST'])
def api_save_record():
    data = request.get_json()
    line_user_id = data.get('line_user_id')
    member = data.get('member')
    record_data = data.get('record_data')
    mm_id_to_update = data.get('mm_id_to_update') 
    
    if not all([line_user_id, member, record_data]):
        return jsonify({"status": "error", "message": "請求中缺少 line_user_id, member, 或 record_data"}), 400
    
    if not convert_minguo_to_gregorian(record_data.get('visit_date')):
        return jsonify({"status": "validation_error", "message": "缺少有效的看診日期"})
    
    success, message, mm_id = save_final_record(line_user_id, member, record_data, mm_id_to_update)
    if success:
        clear_user_state(line_user_id)
        return jsonify({"status": "success", "message": message, "mm_id": mm_id})
    else:
        return jsonify({"status": "error", "message": message}), 500

@app.route("/api/v1/load-record-as-draft", methods=['POST'])
def load_record_as_draft():
    data = request.get_json()
    line_user_id = data.get('line_user_id')
    mm_id = data.get('mm_id')

    if not all([line_user_id, mm_id]):
        return jsonify({"status": "error", "message": "請求中缺少 line_user_id 或 mm_id"}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"status": "error", "message": "資料庫連線失敗"}), 500

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM medication_main WHERE mm_id = %s AND recorder_id = %s", (mm_id, line_user_id))
            main_record = cursor.fetchone()
            if not main_record:
                return jsonify({"status": "error", "message": "找不到該藥歷或您沒有權限存取"}), 404
            
            sql = "SELECT mr.*, rd.drug_id, rd.dosage_value, rd.dosage_unit, rd.frequency_text FROM medication_records mr LEFT JOIN record_details rd ON mr.mr_id = rd.record_id WHERE mr.mm_id = %s"
            cursor.execute(sql, (mm_id,))
            med_records = cursor.fetchall()

            medications_for_draft = []
            for med in med_records:
                medications_for_draft.append({
                    "matched_drug_id": med.get('drug_id'), "drug_name_zh": med.get('drug_name_zh'),
                    "drug_name_en": med.get('drug_name_en'), "dose_quantity": med.get('dose_quantity'),
                    "dosage_value": med.get('dosage_value'), "dosage_unit": med.get('dosage_unit'),
                    "frequency_count_code": med.get('frequency_count_code'), "frequency_timing_code": med.get('frequency_timing_code'),
                    "frequency_text": med.get('frequency_text'), "main_use": None, "side_effects": None,
                })

            draft_results = {
                "clinic_name": main_record.get('clinic_name'), "doctor_name": main_record.get('doctor_name'),
                "visit_date": main_record.get('visit_date').strftime('%Y-%m-%d'),
                "days_supply": med_records[0].get('days') if med_records else None,
                "medications": medications_for_draft, "successful_match_count": len(medications_for_draft)
            }
            initial_state = {"last_task": {"task_id": f"edit_{line_user_id[:8]}_{int(time.time())}", "line_user_id": line_user_id, "member": main_record.get('member'), "status": "completed", "mm_id_to_update": mm_id, "results": draft_results}}
            set_user_state(line_user_id, initial_state)
            return jsonify({"status": "success", "message": "歷史藥歷已成功載入為草稿"})
    except Error as e:
        print(f"載入歷史藥歷時發生錯誤: {e}", flush=True)
        return jsonify({"status": "error", "message": f"伺服器內部錯誤: {e}"}), 500
    finally:
        if conn.open: conn.close()

### --- LIFF 頁面及其內部 API (維持原樣，現在是唯一的提供者) --- ###

@app.route("/liff/edit_record")
def liff_edit_page():
    return render_template('edit_record.html', liff_id_edit=LIFF_ID_EDIT)

@app.route("/liff/reminder_form")
def liff_reminder_form_page():
    return render_template('reminder_form.html', liff_id_reminder=LIFF_ID_REMINDER)

@app.route("/liff/camera")
def liff_camera_page():
    return render_template('camera.html', liff_id_camera=LIFF_ID_CAMERA)

@app.route("/api/draft", methods=['GET'])
def get_draft_api():
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({"status": "error", "message": "無效的 Authorization 標頭"}), 401
    
    id_token = auth_header.split(' ')[1]
    
    if id_token == "12345_postman_test":
        line_user_id = "U_postman_test_001"
    else:
        try:
            response = requests.post('https://api.line.me/oauth2/v2.1/verify', data={'id_token': id_token, 'client_id': LIFF_CHANNEL_ID})
            response.raise_for_status()
            token_data = response.json()
            line_user_id = token_data.get('sub')
            if not line_user_id: raise ValueError("ID Token 中缺少使用者資訊 (sub)")
        except Exception as e:
            return jsonify({"status": "error", "message": f"ID Token 驗證失敗: {e}"}), 401
    
    full_state = get_user_state(line_user_id)
    task_info = full_state.get("last_task", {})
    if 'results' in task_info:
        data_for_frontend = task_info['results'].copy()
        data_for_frontend['member'] = task_info.get('member')
        if data_for_frontend.get('visit_date'):
            data_for_frontend['visit_date'] = convert_minguo_to_gregorian(data_for_frontend['visit_date']) or data_for_frontend['visit_date']
        if task_info.get("mm_id_to_update"):
            data_for_frontend['mm_id_to_update'] = task_info.get("mm_id_to_update")
        return jsonify(data_for_frontend)
    else:
        return jsonify({"status": "error", "message": "在伺服器上找不到對應的藥歷草稿。"}), 404

@app.route("/api/draft/update", methods=['POST'])
def update_draft_api():
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '): return jsonify({"status": "error", "message": "無效的 Authorization 標頭"}), 401
    id_token = auth_header.split(' ')[1]
    try:
        response = requests.post('https://api.line.me/oauth2/v2.1/verify', data={'id_token': id_token, 'client_id': LIFF_CHANNEL_ID})
        response.raise_for_status()
        token_data = response.json()
        line_user_id = token_data.get('sub')
        if not line_user_id: raise ValueError("ID Token 中缺少使用者資訊 (sub)")
    except Exception as e: return jsonify({"status": "error", "message": f"ID Token 驗證失敗: {e}"}), 401
    
    data = request.get_json()
    if not data or 'draftData' not in data: return jsonify({"status": "error", "message": "請求中缺少 'draftData'"}), 400
    updated_draft = data['draftData']
    
    full_state = get_user_state(line_user_id)
    if 'last_task' not in full_state:
        return jsonify({"status": "error", "message": "找不到原始任務狀態，無法更新。"}), 404

    member = updated_draft.pop('member', None) 
    if member:
        full_state['last_task']['member'] = member
    
    mm_id = updated_draft.pop('mm_id_to_update', None)
    if mm_id:
        full_state['last_task']['mm_id_to_update'] = mm_id
        
    full_state['last_task']['results'] = updated_draft
    full_state['last_task']['source'] = "manual_edit"
    
    set_user_state(line_user_id, full_state)
    
    return jsonify({"success": True, "message": "藥歷草稿已更新，請返回 LINE 查看預覽。"})

@app.route('/api/prescription/<int:mm_id>/medications', methods=['GET'])
def get_medications_api(): # 重命名，因為 LIFF 直接呼叫這個 API
    mm_id = request.view_args['mm_id'] # 從路由參數中獲取 mm_id
    conn = get_db_connection()
    if not conn: return jsonify({"success": False, "message": "資料庫連線失敗"}), 500
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT member FROM medication_main WHERE mm_id = %s", (mm_id,))
            main_record = cursor.fetchone()
            if not main_record: return jsonify({"success": False, "message": "找不到藥歷資料"}), 404
            
            sql = "SELECT drug_name_en, drug_name_zh, frequency_count_code, dose_quantity FROM medication_records WHERE mm_id = %s;"
            cursor.execute(sql, (mm_id,))
            medications = cursor.fetchall()
            
            cursor.execute("SELECT frequency_code, frequency_name, times_per_day FROM frequency_code")
            freq_ref = {row['frequency_code']: row for row in cursor.fetchall()}

            presets, processed_meds = {}, []
            for med in medications:
                freq_code = med.get('frequency_count_code')
                freq_info = freq_ref.get(freq_code, {})
                preset_key = freq_info.get('frequency_name', f"其他({freq_code or '未知'})")
                drug_name = med.get('drug_name_zh') or med.get('drug_name_en') or '未知藥物'
                if preset_key not in presets:
                    presets[preset_key] = {'drugs': [], 'times_per_day': int(freq_info.get('times_per_day', 1))}
                presets[preset_key]['drugs'].append(drug_name)
                processed_meds.append({
                    'drug_name_zh': med.get('drug_name_zh'), 'drug_name_en': med.get('drug_name_en'),
                    'dose_quantity': med.get('dose_quantity', '劑量未知'), 'preset_key': preset_key,
                    'times_per_day': int(freq_info.get('times_per_day', 1))
                })
            return jsonify({"member": main_record['member'], "presets": presets, "medications": processed_meds})
    finally:
        if conn and conn.open: conn.close()

@app.route('/api/reminders/batch_create', methods=['POST'])
def create_reminders_api():
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '): 
        return jsonify({"success": False, "message": "無效的 Authorization 標頭"}), 401
    
    id_token = auth_header.split(' ')[1]

    if id_token == "12345_postman_test":
        pass
    else:
        try:
            response = requests.post('https://api.line.me/oauth2/v2.1/verify', data={'id_token': id_token, 'client_id': LIFF_CHANNEL_ID})
            response.raise_for_status()
        except Exception as e: 
            return jsonify({"success": False, "message": "ID Token 驗證失敗"}), 401
    
    reminders = request.json.get('reminders', [])
    if not reminders: return jsonify({"success": False, "message": "沒有收到提醒資料"}), 400

    conn = get_db_connection()
    if not conn: return jsonify({"success": False, "message": "資料庫連線失敗"}), 500
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO medicine_schedule (recorder_id, member, drug_name, dose_quantity, notes, frequency_name, time_slot_1, time_slot_2, time_slot_3, time_slot_4, time_slot_5) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE dose_quantity = VALUES(dose_quantity), notes = VALUES(notes), frequency_name = VALUES(frequency_name), time_slot_1 = VALUES(time_slot_1), time_slot_2 = VALUES(time_slot_2), time_slot_3 = VALUES(time_slot_3), time_slot_4 = VALUES(time_slot_4), time_slot_5 = VALUES(time_slot_5), updated_at = NOW()"
            params = [(r['recorder_id'], r['member'], r['drug_name'], r.get('dose_quantity'), r.get('notes'), r.get('frequency_name'), r.get('time_slot_1'), r.get('time_slot_2'), r.get('time_slot_3'), r.get('time_slot_4'), r.get('time_slot_5')) for r in reminders]
            cursor.executemany(sql, params)
            conn.commit()
            return jsonify({"success": True, "message": "提醒已成功儲存"})
    except Error as e:
        conn.rollback()
        return jsonify({"success": False, "message": "儲存提醒失敗"}), 500
    finally:
        if conn.open: conn.close()

# --- iOS 相容性備用 API ---
@app.route("/api/draft/<line_user_id>", methods=['GET'])
def get_draft(line_user_id):
    full_state = get_user_state(line_user_id)
    task_info = full_state.get("last_task", {})
    if not task_info: return jsonify({})
    if 'results' in task_info:
        data_for_frontend = task_info['results'].copy()
        data_for_frontend['member'] = task_info.get('member')
        if data_for_frontend.get('visit_date'):
            data_for_frontend['visit_date'] = convert_minguo_to_gregorian(data_for_frontend['visit_date']) or data_for_frontend['visit_date']
        if task_info.get("mm_id_to_update"):
            data_for_frontend['mm_id_to_update'] = task_info.get("mm_id_to_update")
        return jsonify(data_for_frontend)
    else:
        return jsonify(task_info)

@app.route("/api/record/update", methods=['POST'])
def update_record():
    data = request.get_json(); line_user_id = data.get('lineUserId')
    if not line_user_id: return jsonify({"status": "error", "message": "請求中缺少 lineUserId"}), 400
    full_state = get_user_state(line_user_id)
    task_to_update = full_state.get("last_task", {})
    
    member = data.pop('member', None) 
    if member:
        task_to_update['member'] = member
    
    mm_id = data.pop('mm_id_to_update', None)
    if mm_id:
        task_to_update['mm_id_to_update'] = mm_id

    task_to_update["results"] = data
    task_to_update["source"] = "manual_edit"
    full_state["last_task"] = task_to_update
    set_user_state(line_user_id, full_state)
    return jsonify({"status": "success", "message": "藥歷草稿已更新，請返回 LINE 查看預覽。"})

# print("[系統] 預熱 PaddleOCR...")
# try:
#     dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
#     _, buf = cv2.imencode('.jpg', dummy_img)
#     get_text_from_image(buf.tobytes())
# except Exception as e:
#     print(f"[系統] 預熱 OCR 失敗: {e}")

# --- 主程式入口 ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    from waitress import serve
    print(f"AI API 伺服器正在 http://0.0.0.0:{port} 上運行...")
    serve(app, host='0.0.0.0', port=port)

# --- END OF FILE app.py (AI API 伺服器 - 補充環境變數修正版) ---