import os
from pathlib import Path, PosixPath
import cv2
import json

img_path = PosixPath('/mnt/afs/share_data/R3/v0.3/data/media/raw_data/9.Calib-RoboticArm/20240711_R3v0.3_9.Calibration-RoboticArm/9.Calibration-RoboticArm/12#/WIN_20240711_11_13_45_Pro.jpg')
# label_paths = Path('/mnt/afs/share_data/R3/v0.3/data')
# label_paths = label_paths.rglob(r'*WIN_20240711_11_13_45_Pro.jpg.json')
# label_path = [item for item in label_paths]
label_path = PosixPath('/mnt/afs/share_data/R3/v0.3/data/annotations/label/9.Calib-RoboticArm/20240711_R3v0.3_9.Calibration-RoboticArm/9.Calibration-RoboticArm/12#/WIN_20240711_11_13_45_Pro.jpg.json')

img = cv2.imread(str(img_path), cv2.COLOR_BGR2RGB)
with open(str(label_path), 'r') as f:
    label = json.load(f)

xs = []
ys = []
for box in label["step_1"]["result"] :
    x = box["x"]
    y = box["y"]
    width = box["width"]
    height = box["height"]
    xs.append(x)
    ys.append(y)

img = cv2.circle(img, (int(xs[0]), int(ys[0])), 20, (0, 255, 0), -1)
img = cv2.circle(img, (int(xs[1]), int(ys[1])), 20, (0, 255, 0), -1)
cv2.imwrite('1.jpg', img)
