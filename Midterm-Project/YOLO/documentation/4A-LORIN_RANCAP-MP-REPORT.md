
# Mid-term Project: Implementing Object Detection on a Dataset

## Overview

This project demonstrates how to implement object detection using a YOLO model (specifically, YOLOv8) on a custom dataset. Object detection is a computer vision technique that allows the model to identify and locate objects within an image. In this project, the YOLO model is trained on a custom dataset to recognize specific objects, then evaluated based on its ability to detect these objects accurately.

### Prerequisites
- `Python 3.7+`
- `Google Colab` (optional, but used here for setup)
- Required Libraries: `roboflow`, `ultralytics`, `cv2`, `matplotlib`, `numpy`, `sklearn`

## Installation and Setup

1. **Install Roboflow**  
   `Roboflow` provides a dataset downloading and management API, which simplifies handling datasets in machine learning projects.

   ```python
   !pip install roboflow
   ```

2. **Download Dataset**  
   Using Roboflow, the dataset is downloaded based on a specific project and version in your Roboflow workspace. The dataset will be structured for the `yolov5` format to be compatible with YOLO models.

   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
   version = project.version(VERSION_NUMBER)
   dataset = version.download("yolov5")
   ```

3. **Download COCO Class Names**  
   The COCO dataset class names are downloaded. These labels represent common object categories (e.g., "person", "car") and are necessary for interpreting the YOLO model’s predictions.

   ```python
   !wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names -O /content/coco.names
   ```

4. **Install Ultralytics YOLO Model**  
   The `ultralytics` package provides an interface for training and using YOLO models. YOLO (You Only Look Once) is a popular model for real-time object detection due to its efficiency and speed.

   ```python
   !pip install ultralytics
   ```

## Loading YOLO Model and Class Names
1. **Load the YOLO Model**  
   YOLOv8 is the latest version of the YOLO series, optimized for both speed and accuracy. This project uses the `yolov8n.pt` (YOLO Nano) pretrained weights on the COCO dataset, which offers a balanced performance for general object detection.

   ```python
   from ultralytics import YOLO
   yolo_model = YOLO('yolov8n.pt')
   ```

2. **Load COCO Class Names**  
   The COCO class names are loaded into a list. Each line in the `coco.names` file represents a class name, and this list will be used to label detected objects.

   ```python
   with open('/content/coco.names', 'r') as f:
       classes = [line.strip() for line in f.readlines()]
   ```

## Image Preprocessing
To improve model performance, images are resized and normalized. This function adjusts image dimensions and scales pixel values.

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
The `detect_objects` function performs object detection on images, predicting bounding boxes and confidence scores for each detected object.

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

- **Bounding Boxes**: Each detected object is surrounded by a rectangle, with coordinates `(x1, y1)` and `(x2, y2)` marking the box edges.
- **Confidence Score**: Displays the likelihood of an object being the predicted class.

## Training the YOLO Model
Using the `train` method, the YOLO model is trained on the downloaded dataset with specific parameters. The model iterates over the dataset multiple times (epochs) to minimize prediction errors.

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
This step assesses the model’s detection ability using precision, recall, and mean Average Precision (mAP) metrics.

- **Precision**: Measures how many detected objects are true positives.
- **Recall**: Measures how many actual objects were detected.
- **mAP@0.5**: Mean Average Precision with IoU threshold of 0.5, which checks detection accuracy.
- **mAP@0.5:0.95**: Evaluates mAP at various IoU thresholds for a comprehensive assessment.

```python
results = yolo_model.val(data='/content/Lettuce/data.yaml')
precision = results.box.p.mean()
recall = results.box.r.mean()
map50 = results.box.map50.mean()
map50_95 = results.box.map.mean()
```

## Speed Evaluation
This block tests the time required for detection on a few sample images, providing insight into real-time detection performance.

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
The visualization displays detected objects and their classes, offering a visual confirmation of model performance.

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

- **Dataset Loading**: The dataset is loaded using the Roboflow API, allowing access to image annotations and labels.
- **Model Loading**: YOLOv8, optimized for speed, is loaded with pretrained weights to boost model training.
- **Image Preprocessing**: Images are resized and normalized to fit the input requirements for YOLO, enhancing prediction accuracy.
- **Object Detection**: Detects objects in images and annotates them with class names and confidence scores.
- **Training**: Model learns object characteristics from the dataset, optimizing weights for better detection.
- **Evaluation**: Model performance is validated using precision, recall, and mAP scores, crucial for real-world applications.
- **Speed and Visualization**: Performance in real-time detection is evaluated and visualized, ensuring the model is fast and interpretable.

