from typing import Any
from ultralytics import YOLO
import numpy as np

class Pipeline():
    def __init__(self, ckpt_path:str) -> None:
        self.model = YOLO(ckpt_path)

    def __call__(self, img: np.ndarray, *args: Any, **kwds: Any) -> Any:
        result = self.model(img)[0]
        boxes = result.boxes  # Boxes object for bounding box outputs
        
        cls = boxes.cls.numpy() # Return the class values of the boxes.
        conf = boxes.conf.numpy() # Return the confidence values of the boxes.
        xywh = boxes.xywh.numpy() # Return the boxes in xywh format (not convert to int).

        return {"cls": cls, "conf": conf, "xywh": xywh}