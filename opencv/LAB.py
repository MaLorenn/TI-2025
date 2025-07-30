import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Mask')
cv2.namedWindow('Original')

# 创建 LAB 滑轨
cv2.createTrackbar('L low', 'Mask', 0, 255, nothing)
cv2.createTrackbar('L high', 'Mask', 255, 255, nothing)
cv2.createTrackbar('A low', 'Mask', 0, 255, nothing)
cv2.createTrackbar('A high', 'Mask', 255, 255, nothing)
cv2.createTrackbar('B low', 'Mask', 0, 255, nothing)
cv2.createTrackbar('B high', 'Mask', 255, 255, nothing)

frame_global = None
lab_global = None

def set_lab_trackbars(lab_val, margin=(20, 10, 10)):
    l, a, b = lab_val
    l_low = max(l - margin[0], 0)
    l_high = min(l + margin[0], 255)
    a_low = max(a - margin[1], 0)
    a_high = min(a + margin[1], 255)
    b_low = max(b - margin[2], 0)
    b_high = min(b + margin[2], 255)

    cv2.setTrackbarPos('L low', 'Mask', l_low)
    cv2.setTrackbarPos('L high', 'Mask', l_high)
    cv2.setTrackbarPos('A low', 'Mask', a_low)
    cv2.setTrackbarPos('A high', 'Mask', a_high)
    cv2.setTrackbarPos('B low', 'Mask', b_low)
    cv2.setTrackbarPos('B high', 'Mask', b_high)

# 鼠标回调
def mouse_callback(event, x, y, flags, param):
    global frame_global, lab_global
    if event == cv2.EVENT_LBUTTONDOWN and frame_global is not None and lab_global is not None:
        bgr = frame_global[y, x]
        lab_pixel = lab_global[y, x]
        print(f'你点击的像素 BGR={bgr}, LAB={lab_pixel}')
        set_lab_trackbars(lab_pixel)
        cv2.circle(frame_global, (x, y), 5, (0, 0, 255), 2)

cv2.setMouseCallback('Original', mouse_callback)

cap = cv2.VideoCapture(0)  # 摄像头编号如有需要可改

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_global = frame.copy()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab_global = lab

    # 滑轨值
    l_low = cv2.getTrackbarPos('L low', 'Mask')
    l_high = cv2.getTrackbarPos('L high', 'Mask')
    a_low = cv2.getTrackbarPos('A low', 'Mask')
    a_high = cv2.getTrackbarPos('A high', 'Mask')
    b_low = cv2.getTrackbarPos('B low', 'Mask')
    b_high = cv2.getTrackbarPos('B high', 'Mask')

    lower = np.array([l_low, a_low, b_low])
    upper = np.array([l_high, a_high, b_high])
    mask = cv2.inRange(lab, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame_global)  # 支持点击取色
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        h_center, w_center = lab.shape[0] // 2, lab.shape[1] // 2
        roi_small = lab[h_center-20:h_center+20, w_center-20:w_center+20]
        mean_lab = cv2.mean(roi_small)[:3]
        mean_lab = tuple(int(c) for c in mean_lab)
        print(f"自动检测到中心主LAB: {mean_lab}")
        set_lab_trackbars(mean_lab)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
