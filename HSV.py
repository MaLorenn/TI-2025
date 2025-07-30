import cv2
import numpy as np

def nothing(x):
    pass

# 固定两个点作为ROI区域
x1, y1 = 236, 191
x2, y2 = 381, 293
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)

cv2.namedWindow('ROI Live')
cv2.namedWindow('Mask')

# 创建 HSV 滑轨
cv2.createTrackbar('H low', 'Mask', 0, 179, nothing)
cv2.createTrackbar('H high', 'Mask', 179, 179, nothing)
cv2.createTrackbar('S low', 'Mask', 0, 255, nothing)
cv2.createTrackbar('S high', 'Mask', 255, 255, nothing)
cv2.createTrackbar('V low', 'Mask', 0, 255, nothing)
cv2.createTrackbar('V high', 'Mask', 255, 255, nothing)

frame_roi_global = None
hsv_roi_global = None

def set_hsv_trackbars(hsv_val, margin=(10, 40, 40)):
    h, s, v = hsv_val
    h_low = max(h - margin[0], 0)
    h_high = min(h + margin[0], 179)
    s_low = max(s - margin[1], 0)
    s_high = min(s + margin[1], 255)
    v_low = max(v - margin[2], 0)
    v_high = min(v + margin[2], 255)

    cv2.setTrackbarPos('H low', 'Mask', h_low)
    cv2.setTrackbarPos('H high', 'Mask', h_high)
    cv2.setTrackbarPos('S low', 'Mask', s_low)
    cv2.setTrackbarPos('S high', 'Mask', s_high)
    cv2.setTrackbarPos('V low', 'Mask', v_low)
    cv2.setTrackbarPos('V high', 'Mask', v_high)

# 鼠标回调：只在ROI小窗口中响应
def mouse_callback(event, x, y, flags, param):
    global frame_roi_global, hsv_roi_global
    if event == cv2.EVENT_LBUTTONDOWN and frame_roi_global is not None and hsv_roi_global is not None:
        bgr = frame_roi_global[y, x]
        hsv_pixel = hsv_roi_global[y, x]
        print(f'你点击的像素(ROI内) BGR={bgr}, HSV={hsv_pixel}')
        set_hsv_trackbars(hsv_pixel)
        cv2.circle(frame_roi_global, (x, y), 5, (0, 0, 255), 2)

cv2.setMouseCallback('ROI Live', mouse_callback)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        continue

    frame_roi_global = roi.copy()
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_roi_global = hsv_roi

    # 滑轨值
    h_low = cv2.getTrackbarPos('H low', 'Mask')
    h_high = cv2.getTrackbarPos('H high', 'Mask')
    s_low = cv2.getTrackbarPos('S low', 'Mask')
    s_high = cv2.getTrackbarPos('S high', 'Mask')
    v_low = cv2.getTrackbarPos('V low', 'Mask')
    v_high = cv2.getTrackbarPos('V high', 'Mask')

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv_roi, lower, upper)
    result = cv2.bitwise_and(roi, roi, mask=mask)

    # ROI二值化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 只显示ROI和处理结果
    cv2.imshow('ROI Live', frame_roi_global)  # 支持点击取色
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    cv2.imshow('Binary', binary)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        h_center, w_center = hsv_roi.shape[0] // 2, hsv_roi.shape[1] // 2
        roi_small = hsv_roi[h_center-20:h_center+20, w_center-20:w_center+20]
        mean_hsv = cv2.mean(roi_small)[:3]
        mean_hsv = tuple(int(c) for c in mean_hsv)
        print(f"自动检测到ROI中心主HSV: {mean_hsv}")
        set_hsv_trackbars(mean_hsv)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
