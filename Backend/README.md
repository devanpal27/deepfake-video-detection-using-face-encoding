# Deepfake Detection Backend

This is the Flask backend for the Deepfake Video Detection project. It accepts a video and a reference image, runs the analysis pipeline, and returns the result to the frontend.

## Features

- Accepts video and image uploads
- Runs deepfake analysis
- Returns:
  - prediction label
  - score
  - threshold
  - processed frames
  - detected frames
  - plot image URL
- Serves generated output images

## Tech Stack

- Python
- Flask
- Flask-CORS
- OpenCV
- dlib
- NumPy
- Matplotlib
  
# Backend setup
## Prerequisites
 Make sure you have:
```
Python 3.10 recommended

pip installed
```
Go to the backend folder
```bash
cd Backend
```
Create a virtual environment
```bash
python3 -m venv venv
```
Activate the virtual environment

For Lunix/macOs
```bash
source venv/bin/activate
```
For windows
```bash
venv\Scripts\Activate.ps1
```
Install dependencies
```bash
pip install -r requirements.txt
```
Run the Backend
```bash
python app.py
```

