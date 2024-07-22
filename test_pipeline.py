import Pipeline
import cv2

# load model
ckpt_path = '/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/runs/detect/train6/weights/best.pt'
model = Pipeline.Pipeline(ckpt_path)

# load image
img_path = '/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/data_processed/yolov8_dataset/images/val/WIN_20240710_17_44_10_Pro.jpg'
img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

# print result
# Format: {"cls": ndarray , "conf": ndarray, "xywh": ndarray}
result = model(img)
print(result)