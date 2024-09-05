# CSST106-CSA-LORIN




https://github.com/user-attachments/assets/3986c96f-62e6-4c56-ace5-a636cc89bd32



### Exploring the Role of Computer Vision and Image Processing in AI

Introduction to Computer Vision and Image Processing

- Computer Vision - Enables AI systems to perceive and understand visual data, performing tasks like image classification, object detection, and motion analysis.
- Image Processing - Crucial for enhancing, manipulating, and refining images, preparing them for accurate AI analysis.  Image processing ensures AI models receive high-quality visual inputs, leading to more precise and reliable outputs, which are essential for applications like autonomous vehicles and medical imaging. Together, they drive innovation across multiple industries, transforming how machines interact with the world.

### Types of image Processing Techniques

- Filtering - Filtering is used to enhance image quality by smoothing, sharpening, or removing noise. Filters operate by modifying pixel values based on their neighbors.
- Edge Detection - Edge detection techniques identify significant changes in intensity within an image, which typically correspond to object boundaries.
- Segmentation - Segmentation divides an image into meaningful regions or objects, making it easier to analyze specific parts of an image.

### AI Application: Medical Imaging
Medical imaging is a crucial AI application, leveraging computer vision and image processing to analyze X-rays, MRIs, and CT scans.
AI enhances the ability to diagnose diseases, monitor patient progress, and plan treatments through detailed image analysis.

### How Image Processing is Used:

- Filtering: Enhances image quality by reducing noise and artifacts.
 Example: Gaussian Blur is used to create clearer images, crucial for accurate diagnosis.
- Segmentation: Isolates and identifies specific regions, such as tumors or organs.
Example: In MRI scans, segmentation distinguishes between healthy tissue and anomalies, aiding early disease detection.
- Edge Detection: Delineates boundaries of structures within images.
Example: Canny Edge Detection helps identify tumor edges or blood vessels, providing precise measurements for treatment planning.

### Challenges 

- Noise and Artifacts: Filtering techniques enhance image clarity, making it easier to detect important details obscured by noise.
- 
- Variability in Image Quality: Image processing standardizes images, ensuring consistent and accurate analysis despite equipment differences or patient movement.
- 
- Complexity of Medical Data: Segmentation and edge detection highlight relevant structures, simplifying the analysis of complex medical images and increasing diagnostic accuracy.

### Image Processing Implementation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Milleys/CSST106-CS4A-LORIN/blob/f0af91f479efa348ac481b5bb544aedd174ced5f/4A_LORIN_MP1.ipynb)
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the MRI image
image = cv2.imread('mri_scan.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Perform simple thresholding
_, segmented_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)

# Apply Canny Edge Detection
edges = cv2.Canny(segmented_image, 50, 150)

# Display results using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Detected Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()

```
![output](https://github.com/user-attachments/assets/e1982231-ea60-4198-8e08-99ccfac77e33)


### Conclusion

Effective image processing is crucial in AI as it enhances the quality of visual data, enabling accurate analysis and interpretation. By applying techniques like filtering, segmentation, and edge detection, AI systems can deliver precise results in various applications, such as medical imaging and autonomous vehicles. This activity really showed how image processing can significantly enhance diagnostic accuracy, automate tasks, and help in making better decisions. It emphasized how crucial these techniques are for advancing AI technologies.

