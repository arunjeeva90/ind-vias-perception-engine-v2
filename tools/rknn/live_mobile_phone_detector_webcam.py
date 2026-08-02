#!/usr/bin/env python3
import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

# Add RKNN Model Zoo helper path
RKNN_MODEL_ZOO_ROOT = "/home/vicharak/Mobility_ADAS/ADVIS/DMS/rknn_model_zoo"
sys.path.append(RKNN_MODEL_ZOO_ROOT)

from py_utils.coco_utils import COCO_test_helper
from py_utils.rknn_executor import RKNN_model_container


OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)

PHONE_CLASS_ID = 67
PHONE_CLASS_NAME = "cell phone"
PHONE_CONF_THRESH = 0.25


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)


def dfl(position):
    # Distribution Focal Loss decode
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)

    y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = y / np.sum(y, axis=2, keepdims=True)

    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    if len(boxes) == 0:
        return None, None, None

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_arr = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_arr[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def extract_phone_detections(boxes, scores, classes, co_helper):
    phones = []
    if boxes is None or scores is None or classes is None:
        return phones

    # map letterboxed boxes back to original frame coordinates
    real_boxes = co_helper.get_real_box(boxes)

    for box, score, cl in zip(real_boxes, scores, classes):
        class_id = int(cl)
        conf = float(score)
        if class_id != PHONE_CLASS_ID:
            continue
        if conf < PHONE_CONF_THRESH:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        phones.append((x1, y1, x2, y2, conf))

    return phones


def draw_phone_overlay(frame, phones, fps, infer_ms):
    for x1, y1, x2, y2, conf in phones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"PHONE {conf:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    status = "PHONE_VISIBLE" if phones else "NO_PHONE"
    cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS {fps:.1f} | infer {infer_ms:.1f} ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/mobile_phone_detector/yolov8n.rknn")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--device_id", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--detect-every", type=int, default=5)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = RKNN_model_container(model_path, args.target, args.device_id)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    co_helper = COCO_test_helper(enable_letter_box=True)

    frame_idx = 0
    last_phones = []
    last_infer_ms = 0.0
    last_time = time.time()
    fps = 0.0

    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed")
                break

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            if frame_idx % max(args.detect_every, 1) == 0:
                img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                t0 = time.time()
                outputs = model.run([img])
                last_infer_ms = (time.time() - t0) * 1000.0

                boxes, classes, scores = yolov8_post_process(outputs)
                last_phones = extract_phone_detections(boxes, scores, classes, co_helper)

            vis = draw_phone_overlay(frame.copy(), last_phones, fps, last_infer_ms)

            if not args.no_display:
                cv2.imshow("RKNN Mobile Phone Detector", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        model.release()


if __name__ == "__main__":
    main()
