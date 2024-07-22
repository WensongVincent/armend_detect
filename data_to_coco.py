import os
import json
import random
import shutil


# /mnt/afs/share_data/R3/v0.3/data/
# ├── media/raw_data/9.Calib-RoboticArm/20240711_R3v0.3_9.Calibration-RoboticArm/9.Calibration-RoboticArm/
# │   ├── 1#/
# │   │   ├── image1.jpg
# │   │   ├── image2.jpg
# |   |   ├── ...
# │   ├── 2#/
# │   │   ├── image1.jpg
# │   │   ├── image2.jpg
# |   |   ├── ...
# |   ├── ...
# ├── annotations/label/9.Calib-RoboticArm/20240711_R3v0.3_9.Calibration-RoboticArm/9.Calibration-RoboticArm/
# │   ├── 1#/
# │   │   ├── image1.jpg.json
# │   │   ├── image2.jpg.json
# |   |   ├── ...
# │   ├── 2#/
# │   │   ├── image1.jpg.json
# │   │   ├── image2.jpg.json
# |   |   ├── ...
# |   ├── ...
# ├── data.yaml


# Path to the original dataset
dataset_path = '/mnt/afs/share_data/R3/v0.3/data/'
image_base_path = os.path.join(dataset_path, 'media/raw_data/9.Calib-RoboticArm/20240711_R3v0.3_9.Calibration-RoboticArm/9.Calibration-RoboticArm')
annotation_base_path = os.path.join(dataset_path, 'annotations/label/9.Calib-RoboticArm/20240711_R3v0.3_9.Calibration-RoboticArm/9.Calibration-RoboticArm')

# Paths for YOLO format dataset
yolo_dataset_path = '/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/data_processed/yolov8_dataset'
yolo_image_train_path = os.path.join(yolo_dataset_path, 'images/train')
yolo_image_val_path = os.path.join(yolo_dataset_path, 'images/val')
yolo_label_train_path = os.path.join(yolo_dataset_path, 'labels/train')
yolo_label_val_path = os.path.join(yolo_dataset_path, 'labels/val')

# Create directories
os.makedirs(yolo_image_train_path, exist_ok=True)
os.makedirs(yolo_image_val_path, exist_ok=True)
os.makedirs(yolo_label_train_path, exist_ok=True)
os.makedirs(yolo_label_val_path, exist_ok=True)

# Classes
classes = {"circle-L": 0, "circle-S": 1, "circle-S-obscured": 2}

def convert_annotation(json_file, img_width, img_height):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for obj in data['step_1']['result']:
        class_id = classes[obj['attribute']]
        x_center = (obj['x'] + obj['width'] / 2) / img_width
        y_center = (obj['y'] + obj['height'] / 2) / img_height
        width = obj['width'] / img_width
        height = obj['height'] / img_height
        annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return annotations

# List all folder names
all_folders = [f"{i}#" for i in range(1, 13)]

# Shuffle and split into training and validation sets (8:2 split)
random.shuffle(all_folders)
train_folders = all_folders[:10]
val_folders = all_folders[10:]

print("Training folders:", train_folders)
print("Validation folders:", val_folders)

for folder in train_folders + val_folders:
    image_folder = os.path.join(image_base_path, folder)
    annotation_folder = os.path.join(annotation_base_path, folder)
    
    is_train = folder in train_folders
    image_dest_folder = yolo_image_train_path if is_train else yolo_image_val_path
    label_dest_folder = yolo_label_train_path if is_train else yolo_label_val_path
    
    for img_file in os.listdir(image_folder):
        if img_file.endswith('.jpg'):
            # Copy image
            src_img_path = os.path.join(image_folder, img_file)
            dst_img_path = os.path.join(image_dest_folder, img_file)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Convert annotation
            json_file = os.path.join(annotation_folder, f"{img_file}.json")
            if os.path.exists(json_file):
                annotations = convert_annotation(json_file, 3840, 2160)
                label_file = os.path.join(label_dest_folder, f"{img_file.replace('.jpg', '.txt')}")
                with open(label_file, 'w') as f:
                    f.write('\n'.join(annotations))

# Create dataset.yaml
yaml_content = f"""
train: {yolo_image_train_path}
val: {yolo_image_val_path}

nc: {len(classes)}
names: {list(classes.keys())}
"""

with open(os.path.join(yolo_dataset_path, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

print("Dataset preparation complete!")
