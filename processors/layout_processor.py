"""
일단 triron 에 대해서 만 구현
"""
import os
import json
import cv2
import numpy as np
import time
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

class LayoutProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('tokenizer')
        self.client = grpcclient.InferenceServerClient(url="0.0.0.0:8888")
        self.model_name = "LAYOUT"

        # 레이블 맵 로드
        with open(os.path.join('tokenizer', 'label_map.json'), 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        self.index_to_label = {v: k for k, v in self.label_map.items()}
        self.y_threshold = 10
    
    def _convert_bbox_format(self, bboxes: list) -> np.ndarray:
        """
        bbox 좌표를 [x1, y1, x4, y4] 형태로 변환하여 numpy array로 반환
        """
        converted_bboxes = np.array([[bbox[0, 0], bbox[0, 1], bbox[2, 0], bbox[2, 1]] for bbox in bboxes], dtype=np.int32)
        return converted_bboxes
    
    def _sort_by_reading_order(self, words: list, boxes: list) -> tuple:
        """
        바운딩 박스 좌표를 기준으로 읽기 순서대로 정렬합니다.
        y좌표 차이가 임계값보다 작으면 같은 줄로 간주하고 x좌표로 정렬합니다.
        """
        # (단어, 박스) 쌍 생성
        word_box_pairs = list(zip(words, boxes))
        
        # y좌표로 그룹화
        y_groups = {}
        for word, box in word_box_pairs:
            y_coord = box[1]  # y1 좌표
            assigned = False
            
            # 기존 그룹과 비교
            for group_y in sorted(y_groups.keys()):
                if abs(y_coord - group_y) <= self.y_threshold:
                    y_groups[group_y].append((word, box))
                    assigned = True
                    break
            
            # 새로운 그룹 생성
            if not assigned:
                y_groups[y_coord] = [(word, box)]
        
        # 각 그룹 내에서 x좌표로 정렬하고, 그룹은 y좌표로 정렬
        sorted_words = []
        sorted_boxes = []
        
        for y_coord in sorted(y_groups.keys()):
            # x좌표로 그룹 내 정렬
            group = sorted(y_groups[y_coord], key=lambda x: x[1][0])  # x1 좌표로 정렬
            
            # 정렬된 결과 추가
            group_words, group_boxes = zip(*group)
            sorted_words.extend(group_words)
            sorted_boxes.extend(group_boxes)
        
        return sorted_words, sorted_boxes
    
    def normalize_bbox(self, bbox: list, width: int, height: int) -> list:
        return[
            int(1000 * bbox[0] / width),
            int(1000 * bbox[1] / height),
            int(1000 * bbox[2] / width),
            int(1000 * bbox[3] / height),
        ]

    def _clean_text(self, text_list: list) -> str:
        """
        텍스트 리스트를 정제하여 하나의 문자열로 만듭니다.
        중복된 단어를 제거하고 적절히 공백을 추가합니다.
        """
        # 중복 제거하면서 순서 유지
        unique_words = []
        seen = set()
        for word in text_list:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        # 특별한 처리가 필요한 경우 (예: 주소, 날짜 등의 포맷)
        text = " ".join(unique_words)
        
        # 주소 형식 정제
        if "동" in text and "호" in text:
            text = text.replace(" 동", "동").replace(" 호", "호")
        elif "동" in text:
            text = text.replace(" 동", "동")
        
        # 날짜 형식 정제
        if "." in text:
            text = text.replace(" .", ".")
        
        return text.strip()
    
    def postprocess(self, logits, words, word_ids):
        # entity별 텍스트 저장
        results = {}
        prev_entity = None
        current_text = []
        seen_words = set()

        for word_idx, pred_idx in enumerate(logits):
            word_id = word_ids[word_idx]
            if word_id is not None:
                entity = self.index_to_label[pred_idx]
                if entity !="O":
                    # 새로운 엔티티 시작
                    if entity != prev_entity:
                        if prev_entity is not None:
                            # 이전 엔티티의 텍스트를 정제하여 저장
                            cleaned_text = self._clean_text(current_text)
                            if cleaned_text:
                                results[prev_entity] = cleaned_text
                        current_text = []
                        seen_words = set()  # 새 엔티티 시작시 seen_words 초기화

                    word = words[word_id]
                    if word not in seen_words:
                        current_text.append(word)
                        seen_words.add(word)
                    prev_entity = entity
        
        # 마지막 엔티티 처리
        if current_text and prev_entity is not None:
            results[prev_entity] = " ".join(current_text)
            
        return results


    def predict(self, images, texts, bboxes):
        start_time = time.time()

        if isinstance(images, np.ndarray):
            images = [images]
            texts = [texts]
            bboxes = [bboxes]
        
        batch_size = len(images)
        all_sorted_texts, all_normalized_boxes, images_list = [], [], []

        for i in range(batch_size):
            sorted_texts, sorted_boxes = self._sort_by_reading_order(texts[i], self._convert_bbox_format(bboxes[i]))
            all_sorted_texts.append(sorted_texts)

            image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            width, height = image.shape[1], image.shape[0]
            normalized_boxes = [self.normalize_bbox(box, width, height) for box in sorted_boxes]
            all_normalized_boxes.append(normalized_boxes)

            resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            img_arr = np.array(resized_image).transpose(2,0,1).astype(np.float32)
            images_list.append(img_arr)
        
        encoding = self.tokenizer(
            all_sorted_texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=512
        )

        batch_bbox = []
        for i in range(batch_size):
            word_ids = encoding.word_ids(batch_index=i)
            sample_bbox = []
            for word_id in word_ids:
                if word_id is None:
                    sample_bbox.append([0, 0, 0, 0])
                else:
                    sample_bbox.append(all_normalized_boxes[i][word_id])
            # 좌표 클리핑
            sample_bbox = [[min(max(coord, 0), 1000) for coord in box] for box in sample_bbox]
            batch_bbox.append(sample_bbox)
        
        input_ids = encoding["input_ids"].astype(np.int32)
        attention_mask = encoding["attention_mask"].astype(np.int32)
        token_type_ids = encoding["token_type_ids"].astype(np.int32)
        bbox_np = np.array(batch_bbox, dtype=np.int32)
        images_np = np.stack(images_list, axis=0)
        
        infer_inputs = []
        infer_inputs.append(grpcclient.InferInput("input_ids", input_ids.shape, "INT32"))
        infer_inputs.append(grpcclient.InferInput("bbox", bbox_np.shape, "INT32"))
        infer_inputs.append(grpcclient.InferInput("image", images_np.shape, "FP32"))
        infer_inputs.append(grpcclient.InferInput("attention_mask", attention_mask.shape, "INT32"))
        infer_inputs.append(grpcclient.InferInput("token_type_ids", token_type_ids.shape, "INT32"))

        infer_inputs[0].set_data_from_numpy(input_ids)
        infer_inputs[1].set_data_from_numpy(bbox_np)
        infer_inputs[2].set_data_from_numpy(images_np)
        infer_inputs[3].set_data_from_numpy(attention_mask)
        infer_inputs[4].set_data_from_numpy(token_type_ids)

        infer_outputs = []
        infer_outputs.append(grpcclient.InferRequestedOutput("output"))

        results = self.client.infer(model_name=self.model_name, inputs=infer_inputs, outputs=infer_outputs)
        logits = results.as_numpy('output').argmax(-1)
        
        results = []
        for i in range(batch_size):
            word_ids = encoding.word_ids(batch_index=i)
            sample_result = self.postprocess(logits[i], all_sorted_texts[i], word_ids)
            results.append(sample_result)
            
        return results[0] if batch_size == 1 else results