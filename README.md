# HCI Computer Vision Experiments

A collection of interactive **Computer Vision experiments developed for a Human-Computer Interaction (HCI) course**.

The project explores how computer vision techniques can be used to analyze real-time camera streams and video files, detect objects based on visual properties, analyze colors, detect motion, and count vehicles.

The experiments were implemented independently using **Python, OpenCV, and NumPy**, with shared utility functions placed in `util.py`.

---

## Overview

This project contains five independent Computer Vision experiments:

| # | Experiment                          | Main Concept                                                     |
| - | ----------------------------------- | ---------------------------------------------------------------- |
| 1 | Yellow Object Detection             | Color segmentation and object counting                           |
| 2 | Fruit Quality Detection             | Color-based quality analysis                                     |
| 3 | Color Detection & Price Calculation | Multi-color object detection and price calculation               |
| 4 | Motion Detection                    | Background subtraction and motion detection                      |
| 5 | Vehicle Counter                     | Motion detection, line-based counting, and red vehicle detection |

Each experiment is implemented as a separate Python script and can be executed independently.

---

## Experiments

### 1. Yellow Object Detection

**File:** `experiment_1_yellow_detection.py`

This experiment detects and counts visible yellow objects in a live camera stream.

The implementation:

* Captures frames from the webcam.
* Converts frames from BGR to HSV.
* Isolates the target yellow color.
* Applies morphological operations to reduce noise.
* Detects object contours.
* Filters objects based on their area.
* Draws bounding boxes around detected objects.
* Displays the number of detected yellow objects.
* Applies coordinate smoothing to reduce bounding-box movement.

**Main techniques:**

* HSV Color Space
* Color Thresholding
* Binary Masks
* Morphological Operations
* Contour Detection
* Bounding Boxes
* Coordinate Smoothing

---

### 2. Fruit Quality Detection

**File:** `experiment_2_fruit_quality.py`

This experiment analyzes a fruit through the camera and attempts to determine whether it is **Good** or **Bad** based on the amount of brown-colored pixels detected within the fruit region.

The implementation:

* Detects the fruit using a predefined HSV range.
* Extracts the detected fruit region.
* Analyzes the region for brown-colored pixels.
* Calculates the percentage of brown pixels.
* Classifies the fruit as:

  * `Good`
  * `Bad`
* Displays the classification directly on the detected object.

A brown-pixel ratio greater than the predefined threshold is considered an indication of a bad product.

**Main techniques:**

* HSV Color Space
* Color Thresholding
* Region of Interest (ROI)
* Pixel Analysis
* Morphological Operations
* Contour Detection
* Rule-Based Classification

> **Note:** This experiment is a color-based demonstration and is not a trained machine-learning model for fruit disease or freshness classification.

---

### 3. Color Detection & Price Calculation

**File:** `experiment_3_color_pricing.py`

This experiment detects products based on their colors and calculates their total price.

Three colors are currently defined:

| Color  | Unit Price |
| ------ | ---------: |
| Yellow |          2 |
| Orange |          1 |
| Green  |          3 |

For every frame, the system:

1. Converts the image to HSV.
2. Creates a color mask for each predefined color.
3. Removes noise using morphological operations.
4. Detects contours.
5. Counts objects that meet the minimum area requirement.
6. Calculates the price for each color.
7. Displays the detected object counts and total price.

The total price is calculated using:

`Number of detected objects × Unit price`

**Main techniques:**

* HSV Color Space
* Multi-color Segmentation
* Morphological Operations
* Contour Detection
* Object Counting
* Rule-Based Price Calculation

---

### 4. Motion Detection

**File:** `experiment_4_motion_detection.py`

This experiment detects moving objects within a video.

The system uses **MOG2 Background Subtraction** to distinguish moving foreground objects from the static background.

The processing pipeline includes:

* Loading a video file.
* Resizing frames.
* Converting frames to grayscale.
* Applying Gaussian Blur.
* Performing background subtraction.
* Thresholding the foreground mask.
* Applying morphological operations.
* Detecting contours.
* Filtering small regions.
* Drawing bounding boxes around detected moving objects.

**Main techniques:**

* Background Subtraction
* MOG2
* Grayscale Conversion
* Gaussian Blur
* Thresholding
* Morphological Operations
* Contour Detection

---

### 5. Vehicle Counter

**File:** `experiment_5_vehicle_counter.py`

This experiment analyzes a road video to detect moving vehicles and count vehicles crossing a predefined virtual line.

The system also attempts to identify vehicles that contain a significant amount of red color.

The processing pipeline includes:

* Background subtraction using MOG2.
* Foreground mask generation.
* Thresholding.
* Morphological operations.
* Contour detection.
* Vehicle region extraction.
* Center-point calculation.
* Virtual line detection.
* Vehicle counting.
* Red color analysis using HSV.
* Displaying total vehicles and red vehicles.

The interface displays:

```text
Cars: X
Red Cars: Y
```

**Main techniques:**

* Background Subtraction
* MOG2
* Contour Detection
* Object Localization
* Line-Based Counting
* HSV Color Analysis
* Red Color Segmentation

---

## Project Structure

```text
HCI-Computer-Vision-Experiments/
│
├── experiments/
│   ├── experiment_1_yellow_detection.py
│   ├── experiment_2_fruit_quality.py
│   ├── experiment_3_color_pricing.py
│   ├── experiment_4_motion_detection.py
│   └── experiment_5_vehicle_counter.py
│
├── util.py
│
├── requirements.txt
│
└── README.md
```

### `experiments/`

Contains the five independent Computer Vision experiments.

### `util.py`

Contains shared utility functions used by the experiments.

For example, Experiment 1 uses the `get_limits()` function to obtain HSV color boundaries.

### `requirements.txt`

Contains the Python dependencies required to run the project.

### `README.md`

Project documentation, setup instructions, experiment descriptions, and technical details.

---

## Technologies

### Programming Language

* Python

### Libraries

* OpenCV
* NumPy

### Computer Vision Techniques

* HSV Color Space
* Color Segmentation
* Thresholding
* Morphological Operations
* Contour Detection
* Region of Interest (ROI)
* Background Subtraction
* MOG2
* Object Counting
* Bounding Boxes
* Basic Rule-Based Classification

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/HCI-Computer-Vision-Experiments.git
```

### 2. Navigate to the project directory

```bash
cd HCI-Computer-Vision-Experiments
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

Each experiment is independent and can be executed separately.

From the project root:

### Experiment 1

```bash
python experiments/experiment_1_yellow_detection.py
```

### Experiment 2

```bash
python experiments/experiment_2_fruit_quality.py
```

### Experiment 3

```bash
python experiments/experiment_3_color_pricing.py
```

### Experiment 4

```bash
python experiments/experiment_4_motion_detection.py
```

### Experiment 5

```bash
python experiments/experiment_5_vehicle_counter.py
```

---

## Camera-Based Experiments

Experiments 1, 2, and 3 currently use the computer's default webcam:

```python
cv2.VideoCapture(0)
```

Make sure that the camera is available and that Python has permission to access it.

---

## Video-Based Experiments

Experiments 4 and 5 currently use video files specified directly in the Python code.

Before running these experiments, update the video path in the corresponding script.

For example:

```python
cap = cv2.VideoCapture("path/to/video.mp4")
```

This makes the experiments compatible with a different computer or video location.

---

## Controls

The camera-based experiments can be stopped by pressing:

```text
Q
```

while the OpenCV window is active.

---

## Important Notes

These experiments were developed as **HCI and Computer Vision demonstrations** rather than production-ready detection systems.

The current implementations use predefined thresholds and rule-based techniques. Their performance can therefore be affected by factors such as:

* Lighting conditions
* Camera quality
* Object colors
* Background complexity
* Object size
* Camera movement
* Video quality
* HSV threshold selection

For example, the fruit-quality experiment uses the proportion of brown pixels as a simple indicator of product quality. It should therefore be considered a demonstration of color-based analysis rather than a medical, agricultural, or commercial quality-control system.

---

## Possible Future Improvements

The project can be extended in several ways:

* Add a graphical web interface.
* Support uploaded videos instead of fixed file paths.
* Allow users to select the camera source.
* Add adjustable HSV thresholds.
* Improve object tracking between frames.
* Replace rule-based classification with machine-learning models.
* Improve vehicle tracking to prevent duplicate counting.
* Add real-time performance metrics such as FPS.
* Save detection results.
* Add support for additional object colors and categories.
* Integrate the experiments into a unified interactive application.

---

## HCI Perspective

The project demonstrates how visual input from cameras and video streams can be transformed into interactive information.

Examples include:

* Real-time object counting.
* Visual feedback through bounding boxes.
* Immediate classification feedback.
* Interactive camera-based detection.
* Visual representation of detected objects.
* Real-time vehicle counting and color identification.

These experiments were developed as part of a **Human-Computer Interaction (HCI)** course to explore the interaction between users and computer vision systems.


