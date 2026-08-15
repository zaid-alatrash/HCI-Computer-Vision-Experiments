# Experiment for detecting multiple product colors and calculating the total price
import cv2
import numpy as np

# ===============================
# Define HSV color ranges and the price for each color
# ===============================
# Use dictionaries to store the properties of each color (lower limit, upper limit, and unit price)
color_ranges = {
    'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255]), 'price': 2},  # Yellow color with a price of 2
    'orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([20, 255, 255]), 'price': 1},  # Orange color with a price of 1
    'green':  {'lower': np.array([40, 50, 50]), 'upper': np.array([90, 255, 255]), 'price': 3},    # Green color with a price of 3
}

# color_ranges = {

#     'yellow': {

#         'lower': np.array([15, 40, 40]),
#         'upper': np.array([40, 255, 255]),
#         'price': 2
#     },

#     'orange': {

#         'lower': np.array([5, 40, 40]),
#         'upper': np.array([22, 255, 255]),
#         'price': 1
#     },

#     'green': {

#         'lower': np.array([35, 30, 30]),
#         'upper': np.array([95, 255, 255]),
#         'price': 3
#     }

# }
# ===============================
# Open the device camera
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to HSV to improve color recognition accuracy and reduce the effect of lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    total_price = 0  # Variable for calculating the total price of all detected products
    counts = {}      # Dictionary for storing the number of detected objects for each color

    # ===============================
    # Loop through each color defined in the list
    # ===============================
    for color_name, props in color_ranges.items():
        # Create a mask that isolates only the current color
        mask = cv2.inRange(hsv, props['lower'], props['upper'])

        # Clean pixel noise from the mask using morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find the boundaries of colored objects in the cleaned mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0 # Counter for the current color
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignore very small colored regions (less than 1000 pixels) to avoid errors
            if area > 1000:
                # Draw a rectangle around the detected object
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                count += 1 # Increment the object counter for this color

        # Store the count and calculate the cost for this color (count × price)
        counts[color_name] = count
        total_price += count * props['price']

    # ===============================
    # Display the final results
    # ===============================
    # Combine the count information (e.g., yellow:2 | green:1) to prepare the display text
    info_text = " | ".join([f"{k}:{v}" for k,v in counts.items()])
    
    # Display the text containing the object counts and total price at the top of the image
    cv2.putText(frame, f"{info_text} | Total Price: {total_price}", 
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Display the final video window
    cv2.imshow("Product Color Detection & Pricing", frame)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera and windows when finished
cap.release()
cv2.destroyAllWindows()
