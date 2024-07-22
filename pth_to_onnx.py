from ultralytics import YOLO

# Load a model
model = YOLO("/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/runs/detect/train6/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")