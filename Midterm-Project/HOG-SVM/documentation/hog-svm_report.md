
# HOG-SVM Documentation

## Import Dataset
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="uhjoV3WNT5LUQzPsvxmG")
project = rf.workspace("test-kqntz").project("marul-mucm2")
version = project.version(2)
dataset = version.download("yolov5")
```

## Import Necessary Libraries
```
!pip install roboflow scikit-image opencv-python-headless

# Import necessary libraries
from roboflow import Roboflow
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import os
import glob

```

## Data Preparation
The dataset was prepared by resizing images, normalizing pixel values, and loading bounding box annotations for object detection.

```python
hog_params = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2'}
window_size = (64, 64)  # HOG window size

def load_annotations(annotations_path, img_path):
    bboxes = []
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    # Processing bounding box coordinates
    # ...
    return img, bboxes
```

## Model Building
For object detection, we used the HOG (Histogram of Oriented Gradients) for feature extraction, combined with an SVM (Support Vector Machine) classifier for object classification.

```python
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, y_train)
```

## Training the Model
The model was trained using the extracted HOG features. The `prepare_data` function helps load images, apply HOG features, and split data into positive (object present) and negative samples.

```python
def prepare_data(dataset_path):
    X, y = [], []
    # Image and label paths
    # ...

    # Process each bounding box
    for bbox in bboxes:
        crop = img[y1:y2, x1:x2]
        hog_features = get_hog_features(crop, hog_params)
        X.append(hog_features)
        y.append(1)  # Label as object class
    # ...

X_train, y_train = prepare_data("/content/dataset")
```

## Testing
The model’s performance was evaluated on a test set to assess detection capabilities, particularly focusing on edge cases where the model may struggle.

```python

y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

## Results 

![hog-result1](https://github.com/user-attachments/assets/75c14c36-269d-451b-9b1a-bb63b2c7e6fe)
![hog-result2](https://github.com/user-attachments/assets/d86c15ee-4c30-42e5-a7f4-1f6f63154dc1)
![hog-result3](https://github.com/user-attachments/assets/ca82a084-be24-43b6-9f96-470fb3049897)



