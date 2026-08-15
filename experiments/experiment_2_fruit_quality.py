# Experiment for detecting whether a product is good or bad based on color analysis
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
last_rects = {}
alpha = 0.3

# Tracking range: combines yellow and brown to ensure the rectangle continues
# surrounding the fruit even if its color changes completely
track_lower = np.array([0, 15, 15])
track_upper = np.array([50, 255, 255])

# Brown color range: used inside the detected rectangle to check for mold or damage
brown_lower = np.array([3, 80, 20])
brown_upper = np.array([16, 255, 120])

while True:
    ret, frame = cap.read()
    if not ret: break

    # Convert the image to HSV to make color isolation more accurate and less affected by lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for tracking the object (includes yellow, orange, and brown shades)
    track_mask = cv2.inRange(hsv, track_lower, track_upper)

    # Clean the mask from noise caused by lighting using morphological operations
    kernel = np.ones((5,5), np.uint8)
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel) # Remove small points
    track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel) # Fill holes inside the object

    # Extract the outer contours of the detected objects
    contours, _ = cv2.findContours(track_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter objects based on area (ignore any object smaller than 2000 pixels)
        if area > 2000:
            current_rects.append(cv2.boundingRect(cnt))

    new_last_rects = {}
    for i, rect in enumerate(current_rects):
        # Apply alpha smoothing to keep the rectangle stable and prevent shaking during movement
        if i in last_rects:
            prev = last_rects[i]
            smoothed = [
                int(prev[0]*(1-alpha)+rect[0]*alpha),
                int(prev[1]*(1-alpha)+rect[1]*alpha),
                int(prev[2]*(1-alpha)+rect[2]*alpha),
                int(prev[3]*(1-alpha)+rect[3]*alpha)
            ]
            new_last_rects[i] = smoothed
        else:
            new_last_rects[i] = list(rect)

        rx, ry, rw, rh = new_last_rects[i]
        
        # Extract only the object region (ROI) to analyze its color independently from the rest of the image
        roi_hsv = hsv[ry:ry+rh, rx:rx+rw]
        
        # Create a mask specifically for the brown color inside the detected object region
        brown_pixel_mask = cv2.inRange(roi_hsv, brown_lower, brown_upper)
        
        # Calculate the number of detected brown pixels
        brown_count = cv2.countNonZero(brown_pixel_mask)
        # Calculate the total area of the detected rectangle
        total_pixels = rw * rh
        # Calculate the percentage of brown color relative to the object size
        brown_ratio = (brown_count / total_pixels) * 100

        # Make the decision: if the brown percentage is greater than 1%, classify the product as bad
        if brown_ratio > 1:  
            status = f"Bad"
            color = (0, 0, 255) # Red color for warning
        else:
            status = "Good"
            color = (0, 255, 0) # Green color for a good product

        # Draw the rectangle and display the status above the detected object
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 3)
        cv2.putText(frame, status, (rx, ry-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Update the memory to use smoothing for the next frame
    last_rects = new_last_rects
    cv2.imshow("Lemon Quality Check", frame)

    # Stop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
