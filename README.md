# Vision Tracker --- Real-Time Object Detection & Tracking

Vision Tracker is a Python computer vision project that uses
**YOLO11n**, **OpenCV**, and **ByteTrack** to detect, track, and count
objects in a live camera feed.

## Features

-   Real-time object detection with YOLO11n
-   Object tracking with ByteTrack and tracking IDs
-   On-screen detection boxes and labels
-   Object-count information
-   Live FPS display
-   Designed to run on a regular computer; a compatible GPU is optional

## Requirements

-   Python 3.11 recommended
-   A webcam for live camera input
-   Internet access on the first run if the YOLO model weights need to
    download
-   Windows, macOS, or Linux

Performance depends on your computer's CPU/GPU and camera resolution.

## Installation

### 1. Get the repository

Clone the repository:

``` bash
git clone https://github.com/rayyannew05-eng/Vision-Tracker.git
cd Vision-Tracker
```

Or use **Code → Download ZIP** on GitHub, then extract the ZIP and open
the extracted folder in a terminal.

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

``` bat
.venv\Scripts\activate.bat
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository includes `requirements.txt`, install its packages:

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the local desktop webcam version, make sure the dependencies include
`ultralytics` and `opencv-python`. Use `opencv-python` rather than
`opencv-python-headless` when the code opens a desktop window with
`cv2.imshow()`.

## Run Vision Tracker

From the project folder, with the virtual environment activated, run:

``` bash
python cv.py
```

If your repository's main script has a different filename, use that
filename instead.

The program should open a camera window and display detection/tracking
results. Press **Q** in the camera window to quit if the script uses `q`
as its quit key.

## Webcam troubleshooting

-   Close other applications that may be using the camera.
-   Allow camera access in your operating system's privacy settings.
-   If the camera does not open, check the camera index in `cv.py`. The
    built-in or primary webcam is often index `0`, but this can vary.
-   On Windows, try the camera in the built-in Camera app first.
-   If you see a PyTorch or DLL error, confirm that your Python version
    and installed PyTorch/Ultralytics packages are compatible, then
    reinstall the dependencies in the active virtual environment.

## Model weights

The project uses the YOLO11n model (`yolo11n.pt`). Ultralytics may
download the weights automatically on first use. You usually do not need
to commit model weights or your `.venv` folder to GitHub.

## Project structure

A typical layout is:

``` text
Vision-Tracker/
├── cv.py
├── requirements.txt
├── README.md
└── .gitignore
```

Your repository may contain additional files.

## Technology stack

-   **Python** --- application logic
-   **Ultralytics YOLO11n** --- object detection
-   **OpenCV** --- camera capture and visual output
-   **ByteTrack** --- object tracking

## Notes

-   The local webcam application runs on the computer where the script
    is launched.
-   A GitHub repository stores and shares the code; it does not run the
    desktop webcam application by itself.
-   A cloud-hosted Streamlit app cannot access a visitor's webcam
    through server-side `cv2.VideoCapture(0)`. Browser-based camera
    access requires a separate implementation.

## License

Add a license file if you want to specify how others may use, modify,
and distribute this project.
