import cv2
import numpy as np
import time
import serial

# ------------------- 参数配置 ------------------------
# 固定ROI区域
x1, y1 = 236, 205
x2, y2 = 381, 307
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)

SERIAL_PORT = '/dev/ttyTHS1'   # Jetson Nano 串口
SERIAL_BAUD = 115200

def nothing(x): pass

def sort_pts(pts):
    pts = sorted(pts, key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return np.array(top + bottom)

def detect_corners(roi):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 9, 10, 10)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    _, binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx_outer = None
    if hierarchy is not None:
        quads = []
        for i, contour in enumerate(contours):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                quads.append((area, approx, hierarchy[0][i][3]))
        quads = sorted(quads, key=lambda x: -x[0])
        for area, approx, parent in quads:
            if parent == -1 and approx_outer is None:
                approx_outer = approx.reshape(-1, 2)
            if approx_outer is not None:
                break
        if approx_outer is not None:
            pts0 = sort_pts(approx_outer)
            return pts0
    return None

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

# 串口初始化
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.5)
if ser.isOpen():
    print(f"串口{SERIAL_PORT}已打开，波特率{SERIAL_BAUD}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -------- 可选可视化 -------
cv2.namedWindow('ROI')
cv2.createTrackbar('Thresh', 'ROI', 230, 255, nothing)
cv2.createTrackbar('Min Area', 'ROI', 2, 20, nothing)

t0 = time.time()
corner_pts = None
corner_sent = False
corners_sent_time = None
last_send_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.getTrackbarPos('Thresh', 'ROI')
    min_area = cv2.getTrackbarPos('Min Area', 'ROI')
    if min_area < 1:
        min_area = 1

    elapsed = time.time() - t0

    # --- 前3秒识别角点 ---
    if elapsed < 3.0:
        roi_disp = roi.copy()
        pts = detect_corners(roi)
        if pts is not None:
            corner_pts = pts.copy()
            for idx, (x, y) in enumerate(pts):
                cv2.circle(roi_disp, (x, y), 5, (255, 0, 0), 2)
                cv2.putText(roi_disp, f"C{idx+1}", (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(roi_disp, f"角点识别中:{elapsed:.1f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow('ROI', roi_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- 3秒后只发送一次角点 ---
    if not corner_sent and corner_pts is not None:
        pts_flat = corner_pts.flatten()
        msg2 = 'a' + ','.join(str(int(i)) for i in pts_flat) + 'z'
        ser.write(msg2.encode('utf-8'))
        print(f"角点已发送: {msg2}")
        corners_sent_time = time.time()
        corner_sent = True

    # --- 等待0.1秒后才循环发激光点 ---
    if corner_sent and corners_sent_time and (time.time() - corners_sent_time < 0.1):
        # 还没到0.1秒，啥也不发，只显示
        cv2.imshow('ROI', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- 0.1秒后循环识别并发送激光点 ---
    now = time.time()
    if now - last_send_time >= 0.2:
        last_send_time = now
        bright_point = get_brightest_point(gray, thresh=thresh, min_area=min_area)
        if bright_point:
            x_roi, y_roi = bright_point
            msg1 = f"a{x_roi},{y_roi}z"
            ser.write(msg1.encode('utf-8'))
            print(f"已发送: {msg1}")

    cv2.imshow('ROI', roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()

