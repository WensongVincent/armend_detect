## Checkpoint
Trained .pth ckpt location: 
/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/runs/detect/train6/weights/best.pt


Exported .Onnx ckpt location: 
/mnt/afs/huwensong/workspace/R3_factory_tool_arm_calib/onnx/best.onnx


## Usage:
```python
From Pipeline import Pipeline
import cv2

# load model
ckpt_path = 'path/to/model/checkpoint.pt'
model = Pipeline(ckpt_path)

# load image
img_path = 'path/to/image.jpg'
img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

# predict
# Result Format: {"cls": ndarray , "conf": ndarray, "xywh": ndarray}
result = model(img)
```

Result of model is: {"cls": ndarray , "conf": ndarray, "xywh": ndarray}

where,

**cls** is box classification, cls.shape is (num_of_boxes, ), 

**conf** is classification confidance, conf.shape is (num_of_boxes, )

**xywh** is box information in [box_center_x, box_center_y, box_width, box_height], xywh.shape is (num_of_boxes, 4)