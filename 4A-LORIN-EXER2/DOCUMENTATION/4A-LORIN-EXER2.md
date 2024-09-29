
# 4A-LORIN-EXER2: Feature Extraction and Matching Using OpenCV

This project demonstrates various feature extraction methods using OpenCV, including SIFT, SURF, ORB, and feature matching techniques. The project is implemented in Python and designed for use in Google Colab.

## Tasks Implemented

1. **SIFT Feature Extraction**: Scale-Invariant Feature Transform (SIFT) is used to extract keypoints and descriptors from an image.
2. **SURF Feature Extraction**: Speeded-Up Robust Features (SURF) extracts keypoints and descriptors from the image. SURF requires OpenCV's contrib module.
3. **ORB Feature Extraction**: ORB (Oriented FAST and Rotated BRIEF) is an efficient alternative to SIFT and SURF.
4. **Feature Matching**: Keypoints are matched between two images using BFMatcher.
5. **Image Alignment with Homography**: Align two images by computing a homography matrix based on the matched features.
6. **Combining Feature Extraction Methods**: Uses both SIFT and ORB to extract features and demonstrate multi-method feature extraction.

## Setup and Installation

Before you begin, ensure that the following dependencies are installed:

```bash
# Install system dependencies
!apt-get update
!apt-get install -y cmake build-essential pkg-config

# Clone OpenCV repositories
!git clone https://github.com/opencv/opencv.git
!git clone https://github.com/opencv/opencv_contrib.git

# Build and install OpenCV with non-free algorithms enabled
!mkdir -p opencv/build
# %cd opencv/build
!cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=OFF ..
!make -j8
!make install
```

## Task 1: SIFT Feature Extraction

SIFT detects and describes local features in images. Here's a snippet for SIFT feature extraction:

```python
import cv2
import matplotlib.pyplot as plt

# Load Image and Convert to Grayscale
image = cv2.imread("/content/sample.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT Detector
sift = cv2.SIFT_create()

# Detect Keypoints and Descriptors
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# Draw Keypoints
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the Image with Keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints")
plt.show()
```

## Task 2: SURF Feature Extraction

SURF is another feature extraction method that requires the contrib module of OpenCV. Here's the code:

```python
surf = cv2.xfeatures2d.SURF_create()

# Detect Keypoints and Descriptors
keypoints, descriptors = surf.detectAndCompute(gray_image, None)
```

## Task 3: ORB Feature Extraction

ORB is an alternative to SIFT and SURF, designed to be fast and efficient.

```python
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray_image, None)
```

## Task 4: Feature Matching

After detecting keypoints and descriptors, you can match them between two images:

```python
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
```

## Task 5: Applications of Feature Matching

Feature matching can be used for image alignment. Here's how homography is applied:

```python
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
result = cv2.warpPerspective(image1, M, (w, h))
```

## Processed Images PDF

You can also save processed images as a PDF:

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('/content/processed_images.pdf') as pdf:
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Keypoints")
    pdf.savefig()
    plt.close()
```

## Output

The processed images, including SIFT, SURF, ORB keypoints, and feature matching results, are compiled into a PDF.

```bash
PDF saved as processed_images.pdf
```

## Conclusion

This project provides an introduction to feature extraction and matching techniques using OpenCV. It includes practical applications such as image alignment and combines different methods to improve robustness.
