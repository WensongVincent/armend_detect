from ultralytics import YOLO
from pathlib import Path
import os

model = YOLO('/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/runs/detect/train6/weights/best.pt')

test_dir_list = ['/mnt/afs/share_data/R3/v0.3/data/media/raw_data/9.Calib-RoboticArm/20240802_R3_RoboticArmCalibrationData',
                 '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/9.Calib-RoboticArm/20240802_R3_RoboticArmCalibrationData_PM',
                 '/mnt/afs/share_data/R3/v0.3/data/media/raw_data/9.Calib-RoboticArm/20240806_R3_RoboticArmCalibrationData']
test_img_path = []
for item in test_dir_list:
    # test_dir = Path('/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/data_processed/yolov8_dataset/images/val')
    test_dir = Path(item)
    test_path = test_dir.rglob(r'*.jpg')
    for img_path in test_path:
        test_img_path.append(str(img_path))
results = model(test_img_path)

# Process results list
i = 0
outdir = '/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/result/0813_1'
os.makedirs(outdir, exist_ok=True)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen

    result.save(filename=f"{outdir}/result{i}.jpg")  # save to disk f'/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/result/{name}'
    i += 1
    # import pdb; pdb.set_trace()