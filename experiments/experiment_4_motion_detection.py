import cv2
import numpy as np

# Load the video file from the specified path
cap = cv2.VideoCapture("C:\\Users\\Zaytona\\Videos\\Captures\\v3.WMV")

# Set up the Background Subtraction engine
# This engine builds a model of the static scene and considers sudden changes as motion
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,       # Number of previous frames used by the engine to understand the background
    varThreshold=120,  # Sensitivity threshold; increasing it makes the system ignore minor camera movements
    detectShadows=False # Disable shadow detection to avoid considering a car's shadow as a moving object
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the video to speed up processing and standardize the display
    frame = cv2.resize(frame, (800, 450))
    
    # Convert the image to grayscale and apply Gaussian Blur
    # This step is essential for reducing digital noise and pixel movement caused by camera motion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    # Apply the engine to the smoothed image to obtain the motion mask
    fg_mask = bg_subtractor.apply(gray)
    
    # Convert the mask to a binary image (black and white) to clearly identify moving objects
    _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

    # Improve the mask using morphological operations
    # MORPH_CLOSE: connects separated parts of the same object (such as connecting the top and bottom of a car)
    # dilate: enlarges the white region to ensure the rectangle surrounds the entire object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    # Find the outer contours of the white regions detected in the mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate the region area; if it is smaller than 3000 pixels, consider it noise (such as tree or sidewalk movement)
        if cv2.contourArea(cnt) < 3000: 
            continue

        # Determine the dimensions of the rectangle surrounding the moving object
        x, y, w, h = cv2.boundingRect(cnt)

        # Draw a green rectangle around the moving object in the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the final window showing rectangles around moving objects only
    cv2.imshow("Motion Detection Only", frame)

    # Exit when the 'q' key is pressed or wait 30 milliseconds between frames
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
