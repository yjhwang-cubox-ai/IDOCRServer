import math
# import torch
import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon
import pyclipper
import tritonclient.grpc as grpcclient

class DBNet:
    def __init__(self, model_path, use_triton=False, triton_url=None):
        self.use_triton = use_triton
        if use_triton:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
            self.model_name = model_path
        else:
            self.session = ort.InferenceSession(model_path)

    def detect_text(self, img: np.ndarray):
        # Resize image
        h, w = img.shape[:2]
        long_side = 1152
        if h > w:
            new_height = long_side
            new_width = int(math.ceil(new_height / h * w / 32) * 32)
        else:
            new_width = long_side
            new_height = int(math.ceil(new_width / w * h / 32) * 32)
        scale_width = new_width / w
        scale_height = new_height / h
        resized_img = cv2.resize(img, (new_width, new_height))

        # Normalize, nchw, batch, ...
        resized_img = resized_img.astype(np.float32)
        RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        resized_img -= RGB_MEAN
        resized_img /= 255
        resized_img = resized_img.transpose(2, 0, 1)
        # Add batch dimension
        x = np.expand_dims(resized_img, axis=0)

         # Inference
        if self.use_triton:
            inputs = []
            inputs.append(grpcclient.InferInput("img", [1, x.shape[1], x.shape[2], x.shape[3]], "FP32"))
            inputs[0].set_data_from_numpy(x)
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput("approximate_binary_map"))
            results = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
            outputs = results.as_numpy("approximate_binary_map")
            approximate_binary_map = outputs[0, 0]
        else:
            # ONNX inference
            outputs = self.session.run(['approximate_binary_map'], {"img": x.numpy()})
            approximate_binary_map = outputs[0, 0]

        # Postprocess
        _, text_score = cv2.threshold(approximate_binary_map, 0.4, 1, 0)

        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8), connectivity=4)
        polygons = []
        for k in range(1, nLabels):
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 50: continue

            segmap = np.zeros(text_score.shape, dtype=np.uint8)
            segmap[labels == k] = 255

            contours, _ = cv2.findContours(segmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) <= 0:
                continue
            contour = contours[0][:, 0, :]

            height = stats[k, cv2.CC_STAT_HEIGHT]
            width = stats[k, cv2.CC_STAT_WIDTH]
            r = 1.80 if width / height < 10 else 3.3
            pg = Polygon(contour)
            d = pg.area * r / pg.length
            dilate_polygon = self.vatti_clipper(contour, d)
            if len(dilate_polygon) <= 0:
                continue
            dilate_polygon = np.array(dilate_polygon, dtype=np.int32)
            rectangle = cv2.minAreaRect(dilate_polygon)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(dilate_polygon[:, 0]), max(dilate_polygon[:, 0])
                t, b = min(dilate_polygon[:, 1]), max(dilate_polygon[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box, dtype=np.float32)
            polygons.append(box)

        for i in range(len(polygons)):
            polygons[i][:, 0] *= 1 / scale_width
            polygons[i][:, 1] *= 1 / scale_height
            polygons[i] = polygons[i].astype(np.int32)

        polygons = self.__sorted_boxes(polygons)
        return polygons


    def __sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        if len(dt_boxes) <= 0:
            return dt_boxes

        # References:
        # https://stackoverflow.com/questions/38654302/how-can-i-sort-contours-from-left-to-right-and-top-to-bottom
        # https://yjs-program.tistory.com/286

        # y 좌표 기준으로 정렬
        y_sorted_boxes = list(sorted(dt_boxes, key=lambda x: np.min(x[:, 1]) ))

        # 각 box 가 몇 번째 줄에 있는지 확인해서 line number 를 배정한다.
        last_box = y_sorted_boxes[0]  # 가장 y 좌표가 작은 bbox
        line = 1
        line_and_box_list = []
        for box in y_sorted_boxes:
            # 서로 다른 줄에 있다고 판단하여 line number 를 증가시킨다.
            if not self.__is_same_line(last_box, box):
                line += 1
            # bbox 에 line number 를 붙인다.
            line_and_box_list.append((line, box))
            last_box = box

        # Line number, x 좌표, y 좌표 순으로 box 를 정렬한다.
        line_sorted_boxes = np.array(
            [x[1] for x in sorted(line_and_box_list, key=lambda x: ( x[0], np.min(x[1][:, 0]), np.min(x[1][:, 1]) ))], dtype=np.int32)
        return line_sorted_boxes


    def __is_same_line(self, box_a, box_b):
        min_y_a, max_y_a = np.min(box_a[:, 1]), np.max(box_a[:, 1])
        min_y_b, max_y_b = np.min(box_b[:, 1]), np.max(box_b[:, 1])

        # 무조건 box_a 가 위에 있다고 가정
        if min_y_a > min_y_b:
            min_y_a, min_y_b = min_y_b, min_y_a
            max_y_a, max_y_b = max_y_b, max_y_a

        if max_y_a <= min_y_b:
            return False

        if max_y_a > min_y_b:
            overlap_ratio = 0.8
            sorted_y = sorted([min_y_b, max_y_b, max_y_a])
            overlap = sorted_y[1] - sorted_y[0]
            return overlap / (max_y_a - min_y_a) >= overlap_ratio or overlap / (max_y_b - min_y_b) >= overlap_ratio

        return False


    def vatti_clipper(self, points, distance):
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        result = np.array(pco.Execute(distance), dtype=object)
        return result if len(result) == 0 else result[0]