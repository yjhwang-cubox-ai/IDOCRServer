# 2024.04.09 모델 배포 샘플 코드 반영
# model: CRAFT

import math
import numpy as np
import cv2
# import torch
import onnxruntime as ort
import tritonclient.grpc as grpcclient

class CRAFT:
    def __init__(self, model_path, use_triton=False, triton_url=None) -> None:
        self.use_triron = use_triton
        if use_triton:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
            self.model_name = model_path
        else:
            self.session = ort.InferenceSession(model_path)

    def detect_text(self, img: np.ndarray):
        # Preprocess
        x, target_ratio = self.resize_aspect_ratio(img, 1080, interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.normalize_mean_variance(x)
        x = x.transpose(2, 0, 1)
        x = np.array([x])
        # x = torch.from_numpy(x)

        # Inference
        if self.use_triron:
            inputs = []
            inputs.append(grpcclient.InferInput("img", [1, x.shape[1], x.shape[2], x.shape[3]], "FP32"))
            inputs[0].set_data_from_numpy(x)
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("output"))
            results = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            output = results.as_numpy("output")
            y = output
        else:
            #ONNX inference
            outputs = self.session.run(None, {'img': x.numpy()})
            y = outputs[0]

        region_score = y[0, :, :, 0]
        affinity_score = y[0, :, :, 1]

        # Postprocess
        dt_boxes = self.__postprocess(region_score, affinity_score, ratio=target_ratio)

        # Sort bboxes left -> right & top -> bottom
        if len(dt_boxes) > 0:
            dt_boxes = self.__sorted_boxes(dt_boxes)
        return dt_boxes


    def resize_aspect_ratio(self, img, square_size, interpolation):
        height, width, channel = img.shape

        # magnify image size
        target_size = square_size

        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc

        return resized, ratio


    def normalize_mean_variance(self, in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        # should be RGB order
        img = in_img.copy().astype(np.float32)

        img -= np.array(
            [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
        )
        img /= np.array(
            [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
            dtype=np.float32,
        )
        return img


    def __postprocess(self, score_text, score_link, ratio=1,
                    text_threshold=0.7, link_threshold=0.5, low_text=0.5
                    ):
        boxes, polys = self.__get_det_boxes(score_text, score_link, text_threshold, link_threshold, low_text)
        ratio_h = ratio_w = 1 / ratio
        dt_boxes = self.__adjust_result_coordinates(boxes, ratio_w, ratio_h)
        return np.array(dt_boxes)


    def __get_det_boxes(self, textmap, linkmap, text_threshold, link_threshold, low_text):
        boxes, labels, mapper = self.__get_det_boxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
        polys = [None] * len(boxes)
        return boxes, polys


    def __get_det_boxes_core(self, textmap, linkmap, text_threshold, link_threshold, low_text):
        # prepare data
        linkmap = linkmap.copy()
        textmap = textmap.copy()
        img_h, img_w = textmap.shape

        """ labeling method """
        ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
        ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

        text_score_comb = np.clip(text_score + link_score, 0, 1)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                            connectivity=4)

        det = []
        mapper = []
        for k in range(1, nLabels):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10: continue

            # thresholding
            if np.max(textmap[labels == k]) < text_threshold: continue

            # make segmentation map
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size.item() * min(w.item(), h.item()) / (
                    w.item() * h.item())) * 2)  # 주의) int32 overflow 발생해서 .item() 으로 native python type 으로 변환함.
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0: sx = 0
            if sy < 0: sy = 0
            if ex >= img_w: ex = img_w
            if ey >= img_h: ey = img_h
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box)

            det.append(box)
            mapper.append(k)

        return det, labels, mapper


    def __adjust_result_coordinates(self, polys, ratio_w, ratio_h, ratio_net=2):
        if len(polys) > 0:
            polys = np.array(polys)
            for k in range(len(polys)):
                if polys[k] is not None:
                    polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
        return polys


    def __sorted_boxes(self, dt_boxes):
        """
            Sort text boxes in order from top to bottom, left to right
            args:
                dt_boxes(array):detected text boxes with shape [4, 2]
            return:
                sorted boxes(array) with shape [4, 2]
            """
        # Reference: https://stackoverflow.com/questions/38654302/how-can-i-sort-contours-from-left-to-right-and-top-to-bottom
        # 최소 높이를 구한다
        min_height = 10000
        for dt_box in dt_boxes:
            height = np.max(dt_box[:, 1]) - np.min(dt_box[:, 1])
            if height < min_height:
                min_height = height

        # y 좌표 기준으로 정렬
        y_sorted_boxes = list(sorted(dt_boxes, key=lambda x: x[0][1]))

        # 각 box 가 몇 번째 줄에 있는지 확인해서 line number 를 배정한다.
        max_y = y_sorted_boxes[0][0][1]  # 가장 y 좌표가 작은 bbox
        line = 1
        line_and_box_list = []
        for box in y_sorted_boxes:
            y = box[0][1]
            # 지금까지 살펴본 y 좌표보다 현재 box 의 y 좌표가 훨씬 (min_height 만큼) 크면
            # 서로 다른 줄에 있다고 판단하여 line number 를 증가시킨다.
            if y >= max_y + min_height:
                max_y = y
                line += 1
            # bbox 에 line number 를 붙인다.
            line_and_box_list.append((line, box))

        # Line number, x 좌표, y 좌표 순으로 box 를 정렬한다.
        line_sorted_boxes = np.array([x[1] for x in sorted(line_and_box_list, key=lambda x: (x[0], x[1][0][0], x[1][0][1]))], dtype=np.int32)
        return line_sorted_boxes

def main():
    model = 'models/dexter.onnx'
    sample = 'driver.jpg'

    detector = DBNet_KR(model_path=model)

    img = cv2.imread(sample)

    dt_boxes = detector.infer(img)

    for idx, box in enumerate(dt_boxes):
        # pt1, pt2, pt3, pt4 = box
        box = box.astype(np.int32)
        img = cv2.polylines(img, [box], True, (0, 255, 0), 2)
        img = cv2.putText(img, str(idx), (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.imwrite("./output.jpg", img)
    print('Please check the output file')

if __name__ == '__main__':
    main()