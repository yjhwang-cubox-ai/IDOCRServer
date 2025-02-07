import unicodedata
import cv2
import numpy as np
from typing import List, Optional, Tuple, Any

class TextProcessor:
    @staticmethod
    def normalize_lao_text(pred_texts: List[str]) -> List[str]:
        """라오어 텍스트 정규화"""
        processed_texts = []
        for pred_text in pred_texts:
            text = unicodedata.normalize("NFKC", pred_text)
            text = TextProcessor._apply_lao_rules(text)
            processed_texts.append(text)
        return processed_texts
    
    @staticmethod
    def _apply_lao_rules(text: str) -> str:
        """라오어 특수 규칙 적용"""
        replacements = {
            'ເເ': 'ແ',
            'ໍໍ': 'ໍ',
            '້ໍ': 'ໍ້',
            'ຫນ': 'ໜ',
            'ຫມ': 'ໝ'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def get_mrz_boxes(dt_boxes: List, image: np.ndarray) -> Tuple[List, List]:
        """MRZ 박스 처리의 메인 함수"""
        text_boxes, mrz_boxes = TextProcessor._get_mrz(dt_boxes, image)
        grouped_mrz_boxes = TextProcessor._grouping_mrz(mrz_boxes)
        splitted_mrz_boxes = TextProcessor._split_mrz(grouped_mrz_boxes, image)
        return text_boxes, splitted_mrz_boxes

    @staticmethod
    def _get_mrz(dt_boxes: List, img: np.ndarray) -> Tuple[List, List]:
        """MRZ 영역과 일반 텍스트 영역 분리"""
        _, img_width, _ = img.shape
        
        text_boxes = []
        mrz_boxes = []
        for dt_box in dt_boxes:
            box_width = dt_box[1][0] - dt_box[0][0]
            if box_width > img_width * 0.5:
                mrz_boxes.append(dt_box)
            else:
                text_boxes.append(dt_box)
                
        return text_boxes, mrz_boxes

    @staticmethod
    def _bbox_center_y(bbox: np.ndarray) -> float:
        """bbox의 중심 y좌표 계산"""
        return np.mean(bbox[:,1])

    @staticmethod
    def _merge_line_bboxes(bboxes: List[np.ndarray]) -> np.ndarray:
        """같은 라인의 bbox들을 하나로 병합"""
        all_points = np.concatenate(bboxes, axis=0)
        x_min = np.min(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        x_max = np.max(all_points[:, 0])
        y_max = np.max(all_points[:, 1])
        return np.array([[x_min, y_min], [x_max, y_min], 
                        [x_max, y_max], [x_min, y_max]], dtype=np.float32)

    @staticmethod
    def _merge_bboxes(bboxes: List[np.ndarray], threshold: int = 10) -> List[np.ndarray]:
        """여러 bbox를 라인별로 병합"""
        bboxes = sorted(bboxes, key=TextProcessor._bbox_center_y)
        merged_bboxes = []
        current_line_bboxes = []
        current_line_center_y = None
        
        for bbox in bboxes:
            center_y = TextProcessor._bbox_center_y(bbox)
            
            if current_line_center_y is None:
                current_line_center_y = center_y
                current_line_bboxes.append(bbox)
            else:
                if abs(center_y - current_line_center_y) <= threshold:
                    current_line_bboxes.append(bbox)
                else:
                    merged_bboxes.append(TextProcessor._merge_line_bboxes(current_line_bboxes))
                    current_line_bboxes = [bbox]
                    current_line_center_y = center_y
        
        if current_line_bboxes:
            merged_bboxes.append(TextProcessor._merge_line_bboxes(current_line_bboxes))
        
        return merged_bboxes

    @staticmethod
    def _grouping_mrz(mrz_points: List) -> List:
        """MRZ 포인트들을 그룹화"""
        if len(mrz_points) > 3:
            bboxes = sorted(mrz_points, key=TextProcessor._bbox_center_y)
            merged_bboxes = TextProcessor._merge_bboxes(bboxes, threshold=10)
            return merged_bboxes
        return mrz_points

    @staticmethod
    def _split_mrz(mrz_boxes: List, image: np.ndarray) -> np.ndarray:
        """MRZ 박스를 문자 단위로 분할"""
        splitted_mrz_boxes = []
        for mrz_box in mrz_boxes:

            left = int(max(0, np.min(mrz_box[:,0])))
            right = int(min(image.shape[0], np.max(mrz_box[:,0])))
            top = int(max(0, np.min(mrz_box[:, 1])))
            bottom = int(max(image.shape[1],np.max(mrz_box[:, 1])))
            
            mrz_img = image[top:bottom, left:right, :].copy()
            height, _, _ = mrz_img.shape
        
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20))
            gray = cv2.cvtColor(mrz_img, cv2.COLOR_BGR2GRAY)
            black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
            black_hat_binary = cv2.threshold(black_hat, 0, 255, 
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            kernel = np.ones((3, 2), np.uint8)
            result = cv2.dilate(black_hat_binary, kernel, iterations=2)
            contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)

            contour_list = [cv2.boundingRect(contour) for contour in contours 
                          if cv2.boundingRect(contour)[3] > height*0.4]
            sorted_data = sorted(contour_list, key=lambda x: x[0])
            
            # 5개의 문자들을 하나의 bbox로 묶기
            box_size = 5
            boxes = []
            for i in range(0, len(contour_list), box_size):
                chunk = sorted_data[i:i + box_size]
                x_min = min(chunk, key=lambda x: x[0])[0]
                y_min = min(chunk, key=lambda x: x[1])[1]
                x_max = max(chunk, key=lambda x: x[0] + x[2])[0] + \
                       max(chunk, key=lambda x: x[0] + x[2])[2]
                y_max = max(chunk, key=lambda x: x[1] + x[3])[1] + \
                       max(chunk, key=lambda x: x[1] + x[3])[3]
                
                padding = 2
                boxes.append([
                    [left+x_min-padding, top+y_min-padding],
                    [left+x_max+padding, top+y_min-padding],
                    [left+x_max+padding, top+y_max+padding],
                    [left+x_min-padding, top+y_max+padding]
                ])
                
            splitted_mrz_boxes.append(boxes)
       
        return np.array(splitted_mrz_boxes, dtype=object)