
# Ego-Motion Prediction with All-Pixel Matching 
## Enhanced Navigation for Blind Guidance

<br>
See our another project for the practical implementation and testing:
<a href="https://github.com/AIS-Clemson/VisionGPT" target="_blank">AIS-Clemson/VisionGPT</a>


<div align="center">
    <img src="./pictures/H_segmentation.jpeg" alt="H-Splitting" style="width: 50%;">
</div>


## Overview
This project introduces an advanced anomaly detection system designed to improve navigation safety for visually impaired individuals and robotic navigation systems. At the heart of this system is an innovative image processing technique that employs an 'H' pattern segmentation, analyzing real-time imagery to accurately identify and categorize potential hazards.

<div align="center">
    <img src="./pictures/HsplitterV2_2.gif" alt="H-Splitting" style="width: 50%;">
</div>


## H-Pattern Segmentation
The 'H' pattern segmentation method divides the captured image into four strategic regions: **Left**, **Right**, **Front**, and **Ground**. This spatial categorization facilitates a nuanced understanding of the environment, enabling the system to focus on areas of interest and ignore irrelevant detections. The segmentation works as follows:

- **Left/Right**: Occupying the outer 25% on either side of the image, these regions are pivotal for detecting moving objects such as vehicles or cyclists that could pose lateral threats.
- **Front**: The central 50% of the image, extending vertically in the upper half, focuses on distant objects directly ahead, aiding in long-range navigation planning.
- **Ground**: This area covers the central 50% widthwise and the lower half vertically, highlighting objects on or near the ground that could present immediate obstacles.


## Image Processing Techniques
We have used several image-processing techniques for our H-splitter:

- **Video Stabilization**: We used SVD to estimate the affine transformation matrix from Feature points extracted in two consecutive frames to counteract camera shake, denoising vector extracted.
- **Vanishing Point Estimation**: Using different techniques to estimate the vanishing point as a reference for segmentation
- **Vanishing Point track analysis** We used a low pass filter over the time axis for further smoothening.
<div align="center">
    <img src="./pictures/HsplitterV2.gif" alt="H-Splitting" style="width: 50%;">
</div>

## Anomaly Detection and Alerts
Anomalies trigger alerts for objects detected in the 'Ground' area or those occupying significant space in the 'Left' or 'Right' regions. By focusing on these critical areas, the system efficiently identifies potential navigation hazards. Alert generation is based on object characteristics such as size (objects occupying >10% of the region), position, and movement patterns, providing users with actionable information.


## Contributions and Feedback
We welcome contributions and feedback to improve the H-Pattern Anomaly Detection system. Please feel free to open issues or submit pull requests with enhancements, bug fixes, or suggestions. Let's work together to make navigation safer for everyone.

### Original contribution:
- Jiayou Qin (Stevens Institute of Technology)
- Hao Wang (Clemson)

### Acknowledgements:

