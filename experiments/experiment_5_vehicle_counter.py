# Code for counting total vehicles and red vehicles
import cv2
import numpy as np

# Load the video from the specified path
cap = cv2.VideoCapture("C:\\Users\\Zaytona\\Videos\\Captures\\v1.WMV")

# Create a Background Subtraction engine to isolate moving objects from the road
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,        # Number of frames remembered by the engine to build the background model
    varThreshold=80,    # Sensitivity threshold (the higher it is, the less noise is detected)
    detectShadows=False # Disable shadow detection to improve processing speed
)

# Define counters and the coordinates of the counting line
car_count = 0
red_car_count = 0
line_position = 225  # Position of the imaginary line that triggers vehicle counting when crossed

detected_ids = set() # Set for storing unique object IDs to prevent duplicate counting
object_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the image to ensure faster processing and standardized dimensions
    frame = cv2.resize(frame, (800,450))

    # Apply the Background Subtraction engine to obtain a mask containing only moving objects
    fg_mask = bg_subtractor.apply(frame)

    # Convert the mask to a binary image (black and white) to remove grayscale variations and noise
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Improve the mask using morphological operations (opening and dilation) to connect parts of the vehicle
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel) # Remove small noise
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)        # Enlarge objects to fill gaps

    # Find the outer contours of each moving object in the mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Ignore very small objects that do not represent vehicles (such as pedestrians or camera noise)
        if cv2.contourArea(cnt) < 2500:
            continue

        # Get the coordinates of the bounding rectangle around the vehicle
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate the vertical center point of the vehicle (Center Y)
        center_y = y + h // 2

        # Check whether the vehicle is currently crossing the imaginary counting line
        if abs(center_y - line_position) < 5:
            # Make sure this vehicle has not already been detected and counted
            if object_id not in detected_ids:
                detected_ids.add(object_id)
                car_count += 1 # Increase the total vehicle counter

                # --- Red color detection logic ---
                # Extract only the vehicle image (ROI) for color analysis
                roi = frame[y:y+h, x:x+w]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Define two red color ranges because red appears at both ends of the HSV scale
                lower_red1, upper_red1 = np.array([0,120,70]), np.array([10,255,255])
                lower_red2, upper_red2 = np.array([170,120,70]), np.array([180,255,255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = mask1 + mask2 # Combine the two masks

                # If the total number of red pixels is large enough, consider it a red vehicle
                if np.sum(red_mask) > 5000:
                    red_car_count += 1

                object_id += 1 # Move to the next ID

        # Draw a green rectangle around every moving vehicle visible in the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the counters directly on the frame
    cv2.putText(frame, "Cars: " + str(car_count), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Red Cars: " + str(red_car_count), (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the final video with the bounding boxes and counters
    cv2.imshow("Vehicle Counter", frame)

    # Exit when the 'q' key is pressed or wait 30 milliseconds between frames
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Close the video source and windows
cap.release()
cv2.destroyAllWindows()
