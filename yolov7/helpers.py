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
from yolov7.utils.datasets import letterbox

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
