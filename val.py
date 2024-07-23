from ultralytics import YOLO

# Load a model
model = YOLO("/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/runs/detect/train6/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/data_processed/yolov8_dataset/dataset.yaml")
# import pdb; pdb.set_trace()