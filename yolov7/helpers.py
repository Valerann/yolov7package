# https://github.com/fcakyon/yolov5-pip/blob/main/yolov5/helpers.py

import sys
import cv2
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
from pathlib import Path
from PIL import Image

from yolov7.coco_labels import coco_class_names
from yolov7.models.common import autoShape
from yolov7.models.experimental import attempt_load
from yolov7.utils.google_utils import attempt_download_from_hub, attempt_download
from yolov7.utils.torch_utils import TracedModel
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box

def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def load_model(model_path, autoshape=True, device='cpu', trace=False, size=640, half=False, hf_model=False):
    """
    Creates a specified YOLOv7 model
    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        trace (bool): if True, model will be traced
        size (int): size of the input image
        half (bool): if True, model will be in half precision
        hf_model (bool): if True, model will be loaded from huggingface hub    
    Returns:
        pytorch model
    (Adapted from yolov7.hubconf.create)
    """
    if hf_model:
        model_file = attempt_download_from_hub(model_path)
    else:
        model_file = attempt_download(model_path)
    
    model = attempt_load(model_file, map_location=device)
    if trace:
        model = TracedModel(model, device, size)

    if autoshape:
        model = autoShape(model)

    if half:
        model.half()

    return model


def detect(yolo7_detector, image, annotate=False):
    try:
        COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        valid_classes = [
            "car",
            "bicycle",
            "motorbike",
            "bus",
            "truck",
            "person",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        ]
        valid_class_ids = [coco_class_names.index(name) for name in valid_classes]
        # Padded resize
        img = letterbox(image, 640, 32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to("cpu")
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = yolo7_detector(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, valid_class_ids, agnostic=False)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
        if annotate:
            for *xyxy, score, classid in reversed(det):
                classid = int(classid)
                score = float(score)
                color = COLORS[int(classid) % len(COLORS)]
                label = "{} : {:f}".format(coco_class_names[classid], score)
                plot_one_box(xyxy, image, label=label, color=color, line_thickness=1)
        return det
    except Exception as e:
        print("Failed running Yolo7", e)
    return []

if __name__ == "__main__":
    model_path = "yolov7.pt"
    device = "cuda:0"
    model = load_model(model_path, device, trace=False, size=640, hf_model=False)
    imgs = [Image.open(x) for x in Path("inference/images").glob("*.jpg")]
    results = model(imgs, size=640, augment=False)
