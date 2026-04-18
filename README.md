# Deepfake Video Detection 

A deepfake detection system that analyzes facial landmark consistency across video frames and compares them with a reference face image to identify manipulated content.

![Project Preview](result.png)

## Overview

This project detects potential deepfake videos by extracting facial landmarks from each frame, normalizing facial geometry, and measuring similarity patterns over time. The system evaluates both:

- similarity between video frames and a reference face image
- similarity between consecutive frames for temporal stability

These signals help identify abnormal facial inconsistencies often seen in manipulated videos.

## Features

- Facial landmark extraction using MediaPipe Face Mesh
- Landmark normalization using eye-based alignment
- Frame-by-frame similarity analysis
- Temporal consistency analysis across consecutive frames
- Smoothed visualization of similarity trends
- Flask backend for API integration
- Ready to connect with a React frontend

## Project Workflow

1. Upload a video and reference image
2. Extract frames from the video
3. Detect facial landmarks in each frame
4. Normalize landmarks for stable comparison
5. Compute:
   - similarity to the reference image
   - similarity to the previous frame
6. Classify the video as REAL / FAKE / Unable to classify
7. Generate an analysis plot

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- Flask

  ## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/devanpal27/Deepfake_video_detection.git
cd Deepfake_video_detection
```
## Project Structure

```text
Deepfake_video_detection/
├── app.py
├── main.py
├── README.md
├── README_image.png
├── requirements.txt
├── .gitignore
├── models/
├── outputs/
├── uploads/
└── src/
    ├── __init__.py
    ├── pipeline.py
    ├── landmarks.py
    ├── metrics.py
    └── visualization.py
