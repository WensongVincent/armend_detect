from ultralytics import YOLO
from pathlib import Path

model = YOLO('/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/runs/detect/train6/weights/best.pt')

test_dir = Path('/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/data_processed/yolov8_dataset/images/val')
test_path = test_dir.rglob(r'*.jpg')
test_img_path = [str(img_path) for img_path in test_path]
results = model(test_img_path)

# Process results list
i = 0
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen

    result.save(filename=f"result{i}.jpg")  # save to disk f'/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/result/{name}'
    i += 1