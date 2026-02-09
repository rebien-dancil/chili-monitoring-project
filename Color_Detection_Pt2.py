import cv2
import numpy as np
import os
from datetime import datetime

cap = cv2.VideoCapture(0)

# Flip camera (optional use only when the camera being used is with phone)
FLIP_CAMERA = True

# Save folder (Windows Pictures)
pictures_folder = os.path.join(os.path.expanduser("~"), "Pictures", "ChiliCaptures")
if not os.path.exists(pictures_folder):
    os.makedirs(pictures_folder)

captured = False  # prevent multiple captures

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FLIP_CAMERA:
        frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # RED (RIPE) range (adjust depending on how deep or rich the color red is)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # GREEN (UNRIPE) range (adjust accordingly on green chili peppers)
    lower_green = np.array([35, 80, 40])
    upper_green = np.array([85, 255, 255])

    # Masks
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ripe_detected = False
    unripe_detected = False

    #RIPE detection (RED)
    for cnt in red_contours:
        if cv2.contourArea(cnt) > 1000:
            ripe_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "RIPE", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #UNRIPE detection (GREEN)
    for cnt in green_contours:
        if cv2.contourArea(cnt) > 1000:
            unripe_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "UNRIPE", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #Capture when RIPE is detected (edit if unripe is also needed to be capture)
    if ripe_detected and not captured:
        captured = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(pictures_folder, f"ripe_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print("📸 RIPE chili detected! Image saved:", filename)
        
    if not ripe_detected:
        captured = False

    cv2.imshow("Chili Ripeness Detection", frame)

    # Exit with Q button
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

