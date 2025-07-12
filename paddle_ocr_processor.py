# --- START OF FILE paddle_ocr_processor.py (繁體中文轉換版) ---

import time
import traceback
from typing import List, Dict, Any, Union
from opencc import OpenCC
from paddleocr import PaddleOCR
import cv2
import numpy as np

# 自訂一個清晰的錯誤類型
class ImageDecodeError(Exception):
    pass

class OCRProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("[PaddleOCR] 正在初始化 PaddleOCR 引擎...", flush=True)
            try:
                cls._instance = super(OCRProcessor, cls).__new__(cls)
                cls._instance.ocr = PaddleOCR(
                    
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=True, # 文檔方向分類model，處理圖片有旋轉的情況
                    use_doc_unwarping=True, # 文本圖像矯正model，修正圖像中文字的傾斜、扭曲等情形
                    use_textline_orientation=True, # 文本行方向model

                    text_line_orientation_model_name = "PP-LCNet_x0_25_textline_ori",

                    text_detection_model_dir = "./PP-OCRv5_mobile_det",
                    text_recognition_model_dir = "./PP-OCRv5_mobile_rec",
                    doc_orientation_classify_model_dir = "./PP-LCNet_x1_0_doc_ori",
                    doc_unwarping_model_dir = "./UVDoc",
                    text_line_orientation_model_dir = "./PP-LCNet_x0_25_textline_ori",

                )
                cls._instance.cc = OpenCC('s2t')
                print("[PaddleOCR] PaddleOCR 引擎初始化完成。", flush=True)
            except Exception as e:
                print(f"[PaddleOCR] 警告：PaddleOCR 引擎初始化失敗！快速模式將不可用。錯誤: {e}", flush=True)
                cls._instance = None
        return cls._instance

    def process_image(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        處理圖片位元組，返回帶有座標和信心分數的結構化文字塊列表。
        不再進行任何形式的行合併。
        """
        if not hasattr(self, 'ocr') or not self.ocr:
            print("[PaddleOCR] 錯誤：OCR 引擎未成功初始化。", flush=True)
            raise RuntimeError("OCR engine is not initialized.")
        
        try:
            print(f"[PaddleOCR] 從記憶體中的圖像位元組開始辨識...", flush=True)
            start_time = time.time()
            img_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (1600,1600))
            if img is None:
                raise ImageDecodeError("[PaddleOCR] 嚴重錯誤: OpenCV 無法解碼記憶體中的圖像位元組。圖片格式可能不受支援或已損壞。")
            
            result = self.ocr.predict(img)

            if not result or not result[0]:
                print("[PaddleOCR] 警告：在圖片中未辨識到任何文字。", flush=True)
                return []
            
            temp = result[0]
            rec_texts = temp["rec_texts"]
            cc = OpenCC('s2t')
            rec_texts = [cc.convert(t) for t in rec_texts]
            rec_box = temp["rec_boxes"]
            confidence = temp["rec_scores"]
            structured_ocr_data = []

            for i in range(len(rec_texts)):
                temp_dic = {
                    "text": rec_texts[i],
                    "box": rec_box[i],
                    "confidence": confidence[i]
                }
                structured_ocr_data.append(temp_dic)

            end_time = time.time()
            elaspe_time = end_time - start_time
            print(f"[OCR]花費時間:{elaspe_time}", flush=True)
            print(f"[PaddleOCR] 辨識完成，返回 {len(structured_ocr_data)} 個文字塊。", flush=True)
            return structured_ocr_data

        except ImageDecodeError as ide:
            print(ide)
            raise
        except Exception as e:
            print(f"[PaddleOCR] 執行 OCR 時發生未知致命錯誤: {e}", flush=True); traceback.print_exc()
            raise

ocr_processor_instance = OCRProcessor()

def get_text_from_image(image_bytes: bytes) -> List[Dict[str, Any]] | None:
    print("[OCR] 執行開始", flush=True)
    if ocr_processor_instance:
        try:
            return ocr_processor_instance.process_image(image_bytes)
        except Exception:
            # 這裡的異常會被 app.py 捕獲並處理為使用者友好的訊息
            return None
    return None

def main():
    img = "./img/004.jpg"
    result = ocr_processor_instance.process_image(img)
    temp = result[0]
    rec_texts = temp["rec_texts"]
    cc = OpenCC('s2t')
    rec_texts = [cc.convert(t) for t in rec_texts]
    rec_box = temp["rec_boxes"]
    confidence = temp["rec_scores"]
    list = []
    for i in range(len(rec_texts)):
        temp_dic = {
            "text": rec_texts[i],
            "box": rec_box[i],
            "confidence": confidence[i]
        }
        list.append(temp_dic)
    
    for i in range(len(list)):
        print(list[i])

if __name__ == "__main__":
    main()
# --- END OF FILE paddle_ocr_processor.py (繁體中文轉換版) ---