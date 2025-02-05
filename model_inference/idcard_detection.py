import argparse

import cv2
import os
import numpy as np
import onnxruntime as ort
from itertools import combinations
import tritonclient.grpc as grpcclient

from typing import Optional

def align_idcard(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    Args:
        img (np.ndarray): HxWxC
        keypoints (np.ndarray): 4x2

    Returns:
        np.ndarray: aligned image
    """
    idcard_ratio = np.array((640, 384))

    dsize_factor = (
        np.sqrt(cv2.contourArea(np.expand_dims(keypoints, 1)))
        / idcard_ratio[0]
    )

    dsize = idcard_ratio * dsize_factor
    dsize = dsize.astype(np.int32)
    dst = np.array(((0, 0), (0, dsize[1]), dsize, (dsize[0], 0)), np.float32)

    M = cv2.getPerspectiveTransform(keypoints.astype(np.float32), dst)
    img = cv2.warpPerspective(img, M, dsize)

    return img


def sort_corner_order(quadrangle: np.ndarray) -> np.ndarray:
    assert quadrangle.shape == (
        4,
        1,
        2,
    ), f"Invalid quadrangle shape: {quadrangle.shape}"

    quadrangle = quadrangle.squeeze(1)
    moments = cv2.moments(quadrangle)
    mcx = round(moments["m10"] / moments["m00"])  # mass center x
    mcy = round(moments["m01"] / moments["m00"])  # mass center y
    keypoints = np.zeros((4, 2), np.int32)
    for point in quadrangle:
        if point[0] < mcx and point[1] < mcy:
            keypoints[0] = point
        elif point[0] < mcx and point[1] > mcy:
            keypoints[1] = point
        elif point[0] > mcx and point[1] > mcy:
            keypoints[2] = point
        elif point[0] > mcx and point[1] < mcy:
            keypoints[3] = point
    return keypoints


def process_quadrangles(quadrangles):
    quad_quadrangles = [
        quad for quad in quadrangles if quad.shape == (4, 1, 2)
    ]
    polygons = [quad for quad in quadrangles if quad.shape != (4, 1, 2)]

    if quad_quadrangles:
        return quad_quadrangles

    if not quad_quadrangles and polygons:
        combinations_of_four = []
        for polygon in polygons:
            combinations_of_four += list(combinations(polygon, 4))

        def calculate_area(coords):
            return cv2.contourArea(
                np.array(coords).reshape((-1, 1, 2)).astype(int)
            )

        # 각 조합에 대해 면적을 계산하고 정렬
        sorted_combinations = sorted(
            combinations_of_four, key=calculate_area, reverse=True
        )

        # 상위 5개의 조합을 선택
        largest_combinations = sorted_combinations[0:5]

        # 두 벡터 사이의 각도를 계산하는 함수
        def angle_between(v1, v2):
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            cos_theta = dot_product / (norm_v1 * norm_v2)
            angle = np.arccos(cos_theta)
            return np.degrees(angle)

        # 각 조합에 대해 직사각형에 가까운 정도를 계산
        def rectness_score(coords):
            coords = np.array(coords).reshape((-1, 2))
            angles = []
            for i in range(4):
                v1 = coords[i] - coords[(i + 1) % 4]
                v2 = coords[(i + 2) % 4] - coords[(i + 1) % 4]
                angle = angle_between(v1, v2)
                angles.append(angle)
            # 직사각형에 가까운 정도는 각도가 90도에 얼마나 가까운지로 판단
            score = sum(abs(angle - 90) for angle in angles)
            return score

        # 직사각형에 가장 가까운 조합
        best_combination = min(largest_combinations, key=rectness_score)
        rect_coords = [
            np.array(best_combination).reshape((-1, 1, 2)).astype(int)
        ]

        return rect_coords


def get_keypoints(
    masks: np.ndarray, morph_ksize=21, contour_thres=0.02, poly_thres=0.03
) -> Optional[np.ndarray]:
    # If multiple masks, select the mask with the largest object.
    if masks.shape[0] > 1:
        masks = masks[
            np.count_nonzero(
                masks.reshape(masks.shape[0], -1), axis=1
            ).argmax()
        ]

    # Post-process mask
    if len(masks.shape) == 3:
        masks = masks.squeeze(0)

    masks = masks.astype(np.uint8)
    # Perform morphological transformation
    masks = cv2.morphologyEx(
        masks,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize)),
    )
    # Find contours (+remove noise)
    contours, _ = cv2.findContours(
        masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        > (masks.shape[0] * masks.shape[1] * contour_thres)
    ]
    # Approximate quadrangles (+remove noise)
    quadrangles = [
        cv2.approxPolyDP(
            contour, cv2.arcLength(contour, True) * poly_thres, True
        )
        for contour in contours
    ]

    rect_coords = process_quadrangles(quadrangles)

    if len(rect_coords) == 1:
        keypoints = sort_corner_order(rect_coords[0])
        return keypoints
    else:

        def calculate_area(box):
            return cv2.contourArea(box)

        largest_box = max(rect_coords, key=calculate_area)
        keypoints = sort_corner_order(largest_box)
        return keypoints

class IDCardDetection:
    """YOLOv8 segmentation model."""

    def __init__(self, model_path, use_triton=False, triton_url=None):
        self.use_triton = use_triton
        
        if use_triton:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
            self.model_name = model_path
            # Triton 서버에서 모델 메타데이터 가져오기
            metadata = self.client.get_model_metadata(model_name=self.model_name)
            # 모델 입력 크기 (높이, 너비) 가져오기
            self.model_height, self.model_width = metadata.inputs[0].shape[2:]
        else:
            # ONNX 모델 로드
            self.session = ort.InferenceSession(model_path)
            # 모델 입력 크기 가져오기
            model_inputs = self.session.get_inputs()[0]
            self.model_height, self.model_width = model_inputs.shape[2:]

        # 모델 입력 데이터 타입 설정 (FP32 사용)
        self.ndtype = np.float32

        self.classes = [
            "kr-idcard",
            "kr-driver",
            "passport",
            "kr-alien-resident",
            "kr-permanent-resident",
            "kr-overseas-resident",
            "vn-cccd-nochip-front",
            "vn-cccd-nochip-back",
            "vn-cccd-chip-front",
            "vn-cccd-chip-back",
            "vn-cmnd-front",
            "vn-cmnd-back",
            "vn-driver-chip-front",
            "vn-driver-chip-back",
            "vn-passport"
        ]

        self.color_palette = [
            (255, 128, 0),
            (255, 153, 51),
            (255, 178, 102),
            (230, 230, 0),
            (0, 255, 255),
            (255, 153, 255),
            (153, 204, 255),
            (255, 102, 255),
            (255, 51, 255),
            (102, 178, 255),
            (51, 153, 255),
            (255, 153, 153),
            (255, 102, 102),
            (255, 51, 51),
            (153, 255, 153),
            (102, 255, 102),
            (51, 255, 51)]

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """

        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # Inference
        if self.use_triton:
            inputs = []
            inputs.append(grpcclient.InferInput("images", [1, 3, im.shape[2], im.shape[3]], "FP32"))
            inputs[0].set_data_from_numpy(im)
            preds = self.client.infer(model_name=self.model_name, inputs=inputs)
        else:
            # ONNX 추론
            preds = self.session.run(None, {self.session.get_inputs()[0].name: im})

        # Post-process
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        return boxes, segments, masks

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        if self.use_triton:
            x, protos = np.array(preds.as_numpy("output0")), np.array(preds.as_numpy("output1"))  # Two outputs: predictions and protos
        else:
            x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            masks_boolean = np.greater(masks, 0.5)
            #masks = np.where(masks > 0.5, 255, 0).astype(np.uint8)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks_boolean)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return masks
        #return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True, file_name=""):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 4], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """

        # Draw rectangles and polygons
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))

            #keypoints = get_keypoints(masks)

            # draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{self.classes[cls_]}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # Show image
        if vis:
            cv2.imshow("demo", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            if not file_name:
                file_name = "demo.jpg"
            cv2.imwrite(file_name, im)
