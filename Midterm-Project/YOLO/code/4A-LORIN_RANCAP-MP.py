# -*- coding: utf-8 -*-
"""4A-LORIN_RANCAP-MP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CmdKWlP2gANT5ndpEwZhBDFPa9fLRazt

**Mid-term Project: Implementing Object Detection on a Dataset**

# **Importing Dataset**
"""

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="uhjoV3WNT5LUQzPsvxmG")
project = rf.workspace("test-kqntz").project("marul-mucm2")
version = project.version(2)
dataset = version.download("yolov5")

"""# Import COCO.NAMES"""

!wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names -O /content/coco.names

"""# Installing Ultralytics for YOLO MODEL"""

!pip install ultralytics

"""# Installing Libraries"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score

"""# Load YOLO MODEL"""

# Install required packages
!pip install -q ultralytics  # If YOLO model comes from ultralytics

# Import libraries
from ultralytics import YOLO
import glob
import os


# Load the YOLO model
yolo_model = YOLO('yolov8n.pt')

# Load class names
with open('/content/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

print("Classes loaded:", classes)

"""# Image Path"""

# Get image paths for the dataset
image_paths = glob.glob('/content/Lettuce/train/images/*.jpg')
print("Found images:", image_paths[:5])  # Display the first 5 image paths

"""# Image Preprocessing"""

# Function to preprocess images
def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Resize and normalize image
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0
    return image_normalized

# Preprocess all images
processed_images = [preprocess_image(path) for path in image_paths]
print("Processed images:", len(processed_images))

"""# Object Detection"""

# Function to detect objects in an image
def detect_objects(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(source=image_rgb)

    print(f"Detected {len(results)} objects in {image_path}")

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()

            if 0 <= int(cls) < len(classes):
                label = f"{classes[int(cls)]}: {conf:.2f}"
            else:
                print(f"Warning: Class ID {int(cls)} not found in coco.names")
                label = f"Class {int(cls)}: {conf:.2f}"

            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image_rgb

"""# Set training parameters and Training Model"""

train_params = {
    "data": "/content/Lettuce/data.yaml",
    "epochs": 10,  # Number of training epochs
    "batch": 16,   # Batch size
    "imgsz": 640,  # Image size for training
}

# Train the model
yolo_model.train(**train_params)

"""# Evaluate the Model"""

# Evaluate the model
results = yolo_model.val(data='/content/Lettuce/data.yaml')
print("Evaluation results:", results)

"""# Calculating Model's Precision, Recall, MAP50, and MAP50_95"""

# Run evaluation
results = yolo_model.val(data='/content/Lettuce/data.yaml')  # Replace with your specific data.yaml path

# Access and calculate mean values for evaluation metrics
precision = results.box.p.mean()  # Mean Precision across classes
recall = results.box.r.mean()     # Mean Recall across classes
map50 = results.box.map50.mean()  # Mean mAP@0.5 across classes
map50_95 = results.box.map.mean()  # Mean mAP@0.5:0.95 across classes

print("Evaluation Results:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"mAP@0.5: {map50:.4f}")
print(f"mAP@0.5:0.95: {map50_95:.4f}")

"""# Speed Evaluation"""

import time

test_paths = glob.glob('/content/Lettuce/test/images/*.jpg')
# Measure speed on sample images
start_time = time.time()
for img_path in test_paths[:4]:
    detected_image = detect_objects(img_path)
end_time = time.time()

print(f'Time taken for detection on {len(test_paths[:4])} images: {end_time - start_time:.2f} seconds')

"""# Object Detection Visualization"""

for img_path in test_paths[:4]:
    detected_image = detect_objects(img_path)
    if detected_image is None:
        continue

    # Display detected image
    plt.figure(figsize=(6, 6))
    plt.imshow(detected_image)
    plt.axis('off')
    plt.title(f'Detected Objects in {os.path.basename(img_path)}')
    plt.show()