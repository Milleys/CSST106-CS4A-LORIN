
# 4A-LORIN-MP3: Feature Extraction, Matching, and Image Alignment

This project demonstrates how to perform feature extraction, feature matching, and image alignment using OpenCV. The project includes the use of SIFT, SURF, and ORB for extracting keypoints and descriptors, followed by brute-force and FLANN-based matching techniques, and concludes with image alignment using homography.

## Steps Implemented

### Step 1: Installation and Setup

The following system dependencies and libraries are required to run the project:

```bash
# Install necessary packages and libraries
!apt-get update
!apt-get install -y cmake build-essential pkg-config

# Clone OpenCV repositories
!git clone https://github.com/opencv/opencv.git
!git clone https://github.com/opencv/opencv_contrib.git

# Build OpenCV with additional modules
!mkdir -p opencv/build
!cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=OFF ..
!make -j8
!make install
```

### Step 2: Load Images

Images are loaded from your local directory or Colab environment for feature extraction and matching:

```python
import cv2
import matplotlib.pyplot as plt

# Load two grayscale images
image1 = cv2.imread('/content/bird1.jpg', 0)
image2 = cv2.imread('/content/bird2.jpg', 0)
```

### Step 3: Feature Extraction Using SIFT, SURF, and ORB

The following methods are used to detect and extract keypoints and descriptors from the images:

#### SIFT Feature Extraction:

```python
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(image1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(image2, None)
```

#### SURF Feature Extraction:

```python
surf = cv2.xfeatures2d.SURF_create()
keypoints1_surf, descriptors1_surf = surf.detectAndCompute(image1, None)
keypoints2_surf, descriptors2_surf = surf.detectAndCompute(image2, None)
```

#### ORB Feature Extraction:

```python
orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(image1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(image2, None)
```

Keypoints detected by SIFT, SURF, and ORB can be visualized using OpenCV's `drawKeypoints()`:

```python
sift_keypoints_image = cv2.drawKeypoints(image1, keypoints1_sift, None)
plt.imshow(cv2.cvtColor(sift_keypoints_image, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.show()
```

### Step 4: Feature Matching

Feature matching is performed using Brute-Force and FLANN methods:

#### Brute-Force Matching with SIFT:

```python
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift_bf = bf.match(descriptors1_sift, descriptors2_sift)
matches_sift_bf = sorted(matches_sift_bf, key=lambda x: x.distance)
```

#### FLANN-based Matching with SIFT:

```python
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_sift_flann = flann.knnMatch(descriptors1_sift, descriptors2_sift, k=2)
```

Visualize matched keypoints:

```python
sift_bf_img = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, matches_sift_bf[:10], None)
plt.imshow(cv2.cvtColor(sift_bf_img, cv2.COLOR_BGR2RGB))
plt.title('SIFT Brute-Force Matches')
plt.show()
```

### Step 5: Image Alignment Using Homography

The matched keypoints can be used to compute a homography matrix for image alignment:

```python
import numpy as np

src_pts = np.float32([keypoints1_sift[m.queryIdx].pt for m in matches_sift_bf]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2_sift[m.trainIdx].pt for m in matches_sift_bf]).reshape(-1, 1, 2)

# Compute the homography matrix
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp image1 to align with image2
aligned_image = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
plt.title('Aligned Image')
plt.show()
```

### Output

All processed images, including keypoints and matched features, are saved and visualized.

### Conclusion

This project demonstrates multiple techniques for feature extraction, feature matching, and image alignment using OpenCV. It provides practical insights into matching keypoints between images and aligning images using homography.
