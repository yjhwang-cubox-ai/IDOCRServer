import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import onnxruntime as ort

class SVTR:
    def __init__(self, model_path, use_triton=False, triton_url=None, dict_path=None):
        self.dict_chars, self.EOS_IDX, self.UKN_IDX = self.read_character_dict(dict_path)
        self.use_triton = use_triton

        if self.use_triton:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
            self.model_name = model_path
        else:
            self.session = ort.InferenceSession(model_path)

    def read_character_dict(self, dict_path):
        
        with open(dict_path, "r", encoding="utf-8") as f:
            texts = f.readlines()
            dict_chars = [text.strip() for text in texts]
        
        # Update dict
        dict_chars = dict_chars + ['<BOS/EOS>', '<UKN>']
        eos_idx = len(dict_chars) - 2
        ukn_idx = len(dict_chars) - 1
        return dict_chars, eos_idx, ukn_idx

    def preprocess(self, img):
        target_height, target_width = 64, 256
        resized_img = cv2.resize(img, (target_width, target_height))
        padding_im = resized_img.astype(np.float32)

        # NHWC to NCHW
        x = np.transpose(resized_img, (2, 0, 1))
        
        # Channel conversion (BGR to RGB)
        x = x[[2, 1, 0], :, :]

        mean = np.array([127.5, 127.5, 127.5]).reshape(3, 1, 1)
        std = np.array([127.5, 127.5, 127.5]).reshape(3, 1, 1)
        x = (x - mean) / std

        return x.astype(np.float32)

    def infer(self, x):
        if self.use_triton:
            inputs = []
            inputs.append(grpcclient.InferInput("input", [x.shape[0], 3, 64, 256], "FP32"))
            inputs[0].set_data_from_numpy(x)
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("output"))
            results = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            output = results.as_numpy("output")
            return output
        else:            
            outputs = self.session.run(None, {"input": x})
            return outputs[0]

    def postprocess(self, pred):
        max_idx = np.argmax(pred, axis=-1)
        max_value = np.max(pred, axis=-1)
        texts = []
        batch_num = pred.shape[0]
        for i in range(batch_num):
            text = ""
            prev_idx = self.EOS_IDX
            for output_score, output_idx in zip(max_value[i], max_idx[i]):
                if output_idx not in (prev_idx, self.EOS_IDX, self.UKN_IDX) and output_score > 0.2:
                    text += self.dict_chars[output_idx]
                    if self.dict_chars[output_idx] == '':
                        text += ' '
                prev_idx = output_idx
            text = text.rstrip()
            texts.append(text)
        return texts

    def recognition(self, imgs):
        preprocessed_batch = np.stack([self.preprocess(img) for img in imgs], axis=0)
        pred = self.infer(preprocessed_batch)
        texts = self.postprocess(pred)
        return texts

    def recognize_texts(self, dt_boxes, img):
        batch_size = 16
        results = []
        
        for idx in range(0, len(dt_boxes), batch_size):
            iter_dt_boxes = dt_boxes[idx:idx+batch_size]
            
            crop_list = []
            for dt_box in iter_dt_boxes:
                left = max(0, int(np.min(dt_box[:, 0])))
                right = min(int(np.max(dt_box[:, 0])), img.shape[1])
                top = max(0, int(np.min(dt_box[:, 1])))
                bottom = min(int(np.max(dt_box[:, 1])), img.shape[0])
                img_crop = img[top:bottom, left:right, :].copy()
                
                crop_list.append(img_crop)
            
            text_batch = self.recognition(crop_list)
            
            results += text_batch
        
        return results