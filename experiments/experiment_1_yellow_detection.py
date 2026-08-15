# Experiment for counting the number of yellow objects
import cv2
import numpy as np
from util import get_limits

# Define the target color in BGR format (yellow)
target_color_bgr = [0, 255, 255] 
cap = cv2.VideoCapture(0)

# Dictionary for storing previous rectangle coordinates for smoothing, and smoothing factor alpha
last_rects = {} 
alpha = 0.3

while True:
    ret, frame = cap.read()
    if not ret: break

    # Convert the image from BGR to HSV because color handling is more accurate in HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Extract the color boundaries (lower and upper values) based on the target color
    lowerLimit, upperLimit = get_limits(color=target_color_bgr)
    
    # Create a mask that shows only the areas containing the target color in white and the rest in black
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Morphological operations to clean the mask by removing small white points and filling gaps inside objects
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill gaps

    # Find the outer boundaries of each separate colored region in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yellow_count = 0 # Counter for yellow objects
    current_rects = [] # List for storing the detected rectangles in the current frame

    for cnt in contours:
        # Calculate the area of the colored region to avoid counting very small points as objects
        area = cv2.contourArea(cnt)
        if area > 1000:
            # Get the coordinates of the bounding rectangle (x, y, width, height)
            x, y, w, h = cv2.boundingRect(cnt)
            current_rects.append((x, y, w, h))
            yellow_count += 1 # Increment the counter when an object meets the conditions

    new_last_rects = {} # Store the new rectangles after smoothing
    for i, rect in enumerate(current_rects):
        # If the object existed in the previous frame, blend the coordinates to reduce movement
        if i in last_rects:
            prev_rect = last_rects[i]
            smoothed = [
                int(prev_rect[0] * (1 - alpha) + rect[0] * alpha),
                int(prev_rect[1] * (1 - alpha) + rect[1] * alpha),
                int(prev_rect[2] * (1 - alpha) + rect[2] * alpha),
                int(prev_rect[3] * (1 - alpha) + rect[3] * alpha)
            ]
            new_last_rects[i] = smoothed
        else:
            new_last_rects[i] = list(rect)

        # Draw a green rectangle around each object using the smoothed coordinates
        rx, ry, rw, rh = new_last_rects[i]
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 4)

    # Update the rectangle list for the next frame
    last_rects = new_last_rects

    # Display the number of detected yellow objects on the screen
    cv2.putText(frame, f"Yellow objects: {yellow_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the final result
    cv2.imshow('Multi-Object Stable Detection', frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
