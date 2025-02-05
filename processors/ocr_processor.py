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

        self.idcard_detection = IDCardDetection(
            model_path=self.args.idcard_model_path,
            use_triton=self.args.use_triton,
            triton_url=triton_url
        )
        if self.args.language == 'kr':
            self.text_detection = CRAFT(
                model_path=self.args.text_det_model_path,
                use_triton=self.args.use_triton,
                triton_url=triton_url
            )
        else:
            self.text_detection = DBNet(
                model_path=self.args.text_det_model_path,
                use_triton=self.args.use_triton,
                triton_url=triton_url
            )
        self.text_recognition = SVTR(
            model_path=self.args.text_recog_model_path,
            use_triton=self.args.use_triton,
            triton_url=triton_url,
            dict_path=self.args.dict_path
        )
    
    def process_single_image(self, image_path: str) -> Optional[Dict]:
        """단일 이미지 처리"""
        total_start_time = time.time()
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        image_name = os.path.basename(image_path)
        
        # ID Card Detection
        start_time = time.time()
        boxes, segments, masks = self.idcard_detection(img)

        if len(boxes) == 0:
            return

        cls = int(boxes[0][5])
        self.logger.log_processing_time('idcard_detection', image_name, time.time() - start_time)
        
        if not boxes.any():
            return None
            
        keypoints = get_keypoints(masks)
        if keypoints is None:
            return None
            
        aligned_img = align_idcard(img=img, keypoints=keypoints)
        
        # Text Detection
        start_time = time.time()
        dt_boxes = self.text_detection.detect_text(aligned_img)
        self.logger.log_processing_time('text_detection', image_name, time.time() - start_time)
        
        text_boxes, mrz_boxes = TextProcessor.get_mrz_boxes(dt_boxes=dt_boxes, image=aligned_img)
        self.logger.log_text_box_count(image_name, len(text_boxes))
        
        # Text Recognition
        start_time = time.time()
        texts = self.text_recognition.recognize_texts(text_boxes, aligned_img)

        if self.args.language == 'lao':
            texts = TextProcessor.normalize_lao_text(texts)

        mrz_texts = []
        if len(mrz_boxes) > 0:
            for mrz_box in mrz_boxes:
                mrz_box = np.array(mrz_box, dtype=np.float32)
                splitted_mrz_texts = self.text_recognition.recognize_texts(mrz_box, aligned_img)
                mrz_text = "".join(splitted_mrz_texts)
                mrz_texts.append(mrz_text)

        self.logger.log_processing_time('text_recognition', image_name, time.time() - start_time)
        
        # 총 처리 시간 기록
        total_time = time.time() - total_start_time
        self.logger.log_processing_time('total_processing', image_name, total_time)
        
        return {
            'aligned_img': aligned_img,
            'text_boxes': text_boxes,
            'texts': texts,
            'mrz_boxes': mrz_boxes,
            'mrz_texts': mrz_texts,
            'class': cls
        }