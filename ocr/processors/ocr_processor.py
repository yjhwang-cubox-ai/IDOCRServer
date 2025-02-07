from model_inference.idcard_detection import IDCardDetection, get_keypoints, align_idcard
from model_inference.text_detection_db import DBNet
from model_inference.text_detection_craft import CRAFT
from model_inference.text_recognition import SVTR
from utils.text_processing import TextProcessor
from processors.time_logger import ProcessingTimeLogger
import time
import cv2
import os
import numpy as np
from datetime import datetime
from typing import Optional, Dict

class OCRProcessor:
    def __init__(self):
        self.setup_models()
    
    def setup_models(self):
        """모델 초기화"""
        triton_url = "0.0.0.0:8888"
        language = 'kr'

        self.idcard_detection = IDCardDetection(
            model_path="KR_IDCARD_DETECTION",
            use_triton=True,
            triton_url=triton_url
        )
        if language == 'kr':
            self.text_detection = CRAFT(
                model_path="KR_TEXT_DETECTION",
                use_triton=True,
                triton_url=triton_url
            )
        else:
            self.text_detection = DBNet(
                model_path="KR_TEXT_DETECTION",
                use_triton=True,
                triton_url=triton_url
            )
        self.text_recognition = SVTR(
            model_path="KR_TEXT_RECOGNITION",
            use_triton=True,
            triton_url=triton_url,
            dict_path='dicts/kr_dict.txt'
        )
    
    def process_single_image(self, image_array: np.ndarray) -> Optional[Dict]:
        """단일 이미지 처리"""
        
        if image_array is None:
            return None
        
        start_time = time.time()
        
        # ID Card Detection
        boxes, segments, masks = self.idcard_detection(image_array)

        if len(boxes) == 0:
            return

        cls = int(boxes[0][5])
        
        if not boxes.any():
            return None
            
        keypoints = get_keypoints(masks)
        if keypoints is None:
            return None
            
        aligned_img = align_idcard(img=image_array, keypoints=keypoints)
        
        # Text Detection
        dt_boxes = self.text_detection.detect_text(aligned_img)
        
        text_boxes, mrz_boxes = TextProcessor.get_mrz_boxes(dt_boxes=dt_boxes, image=aligned_img)
        
        # Text Recognition
        texts = self.text_recognition.recognize_texts(text_boxes, aligned_img)

        mrz_texts = []
        if len(mrz_boxes) > 0:
            for mrz_box in mrz_boxes:
                mrz_box = np.array(mrz_box, dtype=np.float32)
                splitted_mrz_texts = self.text_recognition.recognize_texts(mrz_box, aligned_img)
                mrz_text = "".join(splitted_mrz_texts)
                mrz_texts.append(mrz_text)
        
        return {
            'aligned_img': aligned_img,
            'text_boxes': text_boxes,
            'texts': texts,
            'mrz_boxes': mrz_boxes,
            'mrz_texts': mrz_texts,
            'class': cls
        }