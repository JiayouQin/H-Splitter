
# Anomaly Detection System for Enhanced Navigation Safety

## Overview
This anomaly detection system is designed to enhance navigation safety and situational awareness for visually impaired individuals and other entities requiring navigation assistance, such as robotic systems. Utilizing real-time imagery captured from a camera, the system employs an innovative 'H' pattern segmentation to analyze the environment and identify potential hazards effectively.

## System Description
The core of the system lies in its ability to categorize detected objects within the captured image into four distinct types, based on their spatial location. These categories are aligned with specific segments of the 'H' pattern segmentation, namely:

- **Left**: Objects located on the left 25% of the image. These are typically in motion and may occupy a significant portion of the visual field.
- **Right**: Similar to the Left category but for objects on the right 25% of the image. It's crucial for identifying moving hazards such as vehicles or cyclists.
- **Front**: Focuses on the central 50% of the image's width and the upper half vertically. This region is essential for recognizing objects that are still at a distance but directly ahead, facilitating overall situational assessment and movement planning.
- **Ground**: Occupies the central 50% of the image's width and the lower half vertically, highlighting nearby ground objects. Immediate attention to this area is vital for avoiding close-proximity hazards.

## Data Interpretation
The system records detailed information for each detected object, including its classification, size, and position, with all measurements expressed in percentage terms relative to the image dimensions. This standardized approach ensures better compatibility and interpretation by Large Language Models (LLMs).

## Anomaly Alerts
Anomalies trigger alerts for objects appearing in the 'Ground' area or occupying significant space (greater than 10% in this study) in either the 'Left' or 'Right' regions. This feature is critical for preemptively identifying and responding to potential navigation hazards.

## Structured Data for LLM
Detection and movement data undergo post-processing to be structured in a format conducive to LLM analysis. This step enhances the system's ability to generate accurate and contextually relevant alerts, further improving navigation safety for users.

## Conclusion
This anomaly detection system represents a significant advancement in accessible technology, offering a new level of environmental awareness and safety for visually impaired individuals and automated navigation systems alike.
