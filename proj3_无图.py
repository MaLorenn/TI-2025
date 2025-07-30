import cv2
import numpy as np

# 固定ROI区域
x1, y1 = 238, 196
x2, y2 = 385, 295
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)

def get_brightest_point(gray, thresh=230, min_area=2):
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    bright_point = get_brightest_point(gray, thresh=230, min_area=2)
    if bright_point:
        x_roi, y_roi = bright_point
        x_full, y_full = x_roi + xmin, y_roi + ymin
        print(f"a{x_roi},{y_roi}z")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
