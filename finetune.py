from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms, models

class ArmCalibDetect (nn.Module):
    def __init__(self, ckpt_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = YOLO(ckpt_path)

# model = YOLO('/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/ckpts/yolov8n.pt')
model = YOLO('yolov8n.pt')
results = model.train(model='yolov8n.pt', data='/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/data_processed/yolov8_dataset/dataset.yaml', epochs=10, imgsz=640)
