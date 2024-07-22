from ultralytics import YOLO

# Load a model
model = YOLO("/home/SENSETIME/huwensong/workspace/r3_factory_tool_arm_calib/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")