import cv2
import numpy as np

# 固定ROI区域
x1, y1 = 236, 205
x2, y2 = 381, 307
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)

def nothing(x):
    pass

def get_brightest_point(gray, thresh=228, min_area=2):
    _, mask = cv2.threshold(gray, thresh, 2, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, mask
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None, mask
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy), mask

cv2.namedWindow('Bright Mask')
cv2.namedWindow('ROI')

# 增加二值化阈值和最小面积滑轨
cv2.createTrackbar('Thresh', 'Bright Mask', 230, 255, nothing)
cv2.createTrackbar('Min Area', 'Bright Mask', 2, 20, nothing)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 获取滑轨的当前值
    thresh = cv2.getTrackbarPos('Thresh', 'Bright Mask')
    min_area = cv2.getTrackbarPos('Min Area', 'Bright Mask')
    if min_area < 1:  # 防止为0
        min_area = 1

    bright_point, mask = get_brightest_point(gray, thresh=thresh, min_area=min_area)

    roi_disp = roi.copy()
    if bright_point:
        cv2.circle(roi_disp, bright_point, 7, (0, 255, 0), 2)
        x_roi, y_roi = bright_point
        x_full, y_full = x_roi + xmin, y_roi + ymin
        print(f"激光点ROI内坐标: ({x_roi}, {y_roi})，原图坐标: ({x_full}, {y_full})")
        cv2.putText(roi_disp, f"({x_roi},{y_roi})", (x_roi+10, y_roi),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('ROI', roi_disp)
    cv2.imshow('Bright Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
