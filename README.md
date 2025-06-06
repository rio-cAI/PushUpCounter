PushUpCounter is a simple application that leverages Mediapipe and OpenCV to automatically count push-ups and help maintain proper form for optimal results.

Reference repo: https://github.com/aryanvij02/PushUpCounter

## Project Structure

- **BasicPoseModule.py**  
    This module provides a reusable class for pose detection using Mediapipe. You can integrate `BasicPoseModule` into your own projects and adjust its variables as needed. It exposes key pose landmarks and utilities for analyzing body position. This is a simplified version of `PoseModule.py`.

- **PoseModule.py**  
    An enhanced version of `BasicPoseModule.py`, this module offers more detailed pose detection capabilities. It includes functionalities for angle calculation between body parts, which is crucial for form analysis.

- **PushUpCounter.py**  
    The main script that uses `PoseModule` to detect push-ups in real time from a webcam feed. It counts repetitions and provides feedback on form based on angle calculations provided by `PoseModule.py`.

- **requirements.txt**  
    Lists all Python dependencies required to run the project.

## Usage

1. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
2. Run the main script:
     ```bash
     python PushUpCounter.py
     ```

You can modify or extend any of the scripts to fit your own use case.