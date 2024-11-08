
# 4A-LORIN-MP5.ipynb
---

Automatically generated by Colab.

Original file is located at:  
https://colab.research.google.com/drive/1M-pgRupjFqxA1G7dVpl8X3cRGY-8X_zi

## Machine Problem: Object Detection and Recognition using YOLO

This script demonstrates the use of the YOLO (You Only Look Once) object detection model to recognize objects in images using the YOLOv3 pre-trained model. The code provided detects objects from multiple images and displays bounding boxes with labels on detected objects.

### Table of Contents
1. [Import Libraries](#import-libraries)
2. [Load YOLO Model](#load-yolo-model)
3. [Image Input](#image-input)
4. [Object Detection](#object-detection)
5. [Downloads](#downloads)

---

### 1. Import Libraries

The following libraries are imported to facilitate the object detection process:

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
```

- **cv2**: OpenCV library for image processing and computer vision tasks.
- **numpy**: Library for handling arrays, used here for numerical operations.
- **cv2_imshow**: Colab-specific function to display images in the notebook.

---

### 2. Load YOLO Model

This section loads the YOLO model configuration, pre-trained weights, and COCO class labels.

```python
# Load class labels from coco.names
with open('/content/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet('/content/yolov3.weights', '/content/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

- **coco.names**: File containing COCO dataset class names.
- **yolov3.weights**: Pre-trained YOLO model weights.
- **yolov3.cfg**: Model configuration file defining network structure.
- **output_layers**: Names of YOLO layers from which outputs will be taken for detection.

---

### 3. Image Input

Paths to three images are specified for object detection.

```python
image_paths = ['/content/img1.jpg', '/content/img2.jpg', '/content/img3.jpg']
```

- **image_paths**: List containing the file paths of the images to be processed.

---

### 4. Object Detection

The main detection loop reads each image, processes it through the YOLO model, and draws bounding boxes with labels for detected objects.

```python
# Loop through each image
for image_path in image_paths:
    # Load image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Prepare the image as input for the model
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get label and draw bounding box with label text
                label = str(classes[class_id])
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image with detections
    print(f"Detections for {image_path}:")
    cv2_imshow(img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
```

Explanation:
- **blobFromImage**: Converts image to a format YOLO model can process.
- **cv2.rectangle**: Draws bounding box around detected object.
- **cv2.putText**: Adds object label above bounding box.

---

### 5. Downloads

The required files for YOLO model are downloaded here:

```bash
!wget https://pjreddie.com/media/files/yolov3.weights
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

- **yolov3.weights**: Pre-trained model weights.
- **yolov3.cfg**: Model configuration file.
- **coco.names**: COCO class names for labeling objects.
