
# Mid-term Project: Implementing Object Detection on a Dataset

## Overview

This project demonstrates how to implement object detection using a YOLO model (specifically, YOLOv8) on a custom dataset. The workflow involves setting up the dataset, loading class names, preprocessing images, training the model, and evaluating its performance.

### Prerequisites
- `Python 3.7+`
- `Google Colab` (optional, but used here for setup)
- Required Libraries: `roboflow`, `ultralytics`, `cv2`, `matplotlib`, `numpy`, `sklearn`

## Installation and Setup

1. **Install Roboflow**  
   ```python
   !pip install roboflow
   ```

2. **Download Dataset**  
   Download your dataset from Roboflow using your API key.  
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
   version = project.version(VERSION_NUMBER)
   dataset = version.download("yolov5")
   ```

3. **Download COCO Class Names**  
   COCO class names are required for labeling detected objects.  
   ```python
   !wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names -O /content/coco.names
   ```

4. **Install Ultralytics YOLO Model**  
   The `ultralytics` package provides an interface for YOLO models.  
   ```python
   !pip install ultralytics
   ```

## Loading YOLO Model and Class Names
1. **Load the YOLO Model**  
   Load the YOLO model pretrained on COCO dataset.  
   ```python
   from ultralytics import YOLO
   yolo_model = YOLO('yolov8n.pt')
   ```

2. **Load COCO Class Names**  
   Load the COCO class names for labeling detected objects.  
   ```python
   with open('/content/coco.names', 'r') as f:
       classes = [line.strip() for line in f.readlines()]
   ```

## Image Preprocessing
This project uses a function to resize and normalize images for the YOLO model.

```python
def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0
    return image_normalized
```

## Object Detection
The function `detect_objects` performs object detection and annotates detected objects.

```python
def detect_objects(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(source=image_rgb)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if 0 <= int(cls) < len(classes):
                label = f"{classes[int(cls)]}: {conf:.2f}"
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image_rgb
```

## Training the YOLO Model
The following parameters are used to train the YOLO model.

```python
train_params = {
    "data": "/content/Lettuce/data.yaml",
    "epochs": 10,
    "batch": 16,
    "imgsz": 640,
}
yolo_model.train(**train_params)
```

## Model Evaluation
Evaluate the model’s performance using precision, recall, mAP@0.5, and mAP@0.5:0.95.

```python
results = yolo_model.val(data='/content/Lettuce/data.yaml')
precision = results.box.p.mean()
recall = results.box.r.mean()
map50 = results.box.map50.mean()
map50_95 = results.box.map.mean()
```

## Speed Evaluation
Calculate the time taken to detect objects on test images.

```python
import time
test_paths = glob.glob('/content/Lettuce/test/images/*.jpg')
start_time = time.time()
for img_path in test_paths[:4]:
    detected_image = detect_objects(img_path)
end_time = time.time()
print(f'Time taken for detection on 4 images: {end_time - start_time:.2f} seconds')
```

## Visualization of Detection Results
Display detected objects on images.

```python
for img_path in test_paths[:4]:
    detected_image = detect_objects(img_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(detected_image)
    plt.axis('off')
    plt.title(f'Detected Objects in {os.path.basename(img_path)}')
    plt.show()
```

## Explanation

- **Dataset Loading**: The dataset is loaded using the Roboflow API.
- **Model Loading**: YOLOv8 is used for object detection.
- **Image Preprocessing**: Images are resized and normalized.
- **Object Detection**: The model identifies objects in images, marking them with bounding boxes.
- **Training**: The model trains on the dataset using specified parameters.
- **Evaluation**: The model's performance is evaluated using various metrics.
- **Speed and Visualization**: Speed of detection is tested, and results are visualized.

