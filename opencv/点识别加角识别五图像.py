import cv2
import numpy as np
import time
import serial

# ----------参数区----------
x1, y1 = 236, 205
x2, y2 = 381, 307
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)
SERIAL_PORT = '/dev/ttyTHS1'   # 串口号
SERIAL_BAUD = 9600
THRESH = 230                  # 激光点二值化阈值
MIN_AREA = 2                  # 激光点最小面积
CORNER_BIN_THRESH = 90        # 角点检测二值化阈值

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
    _, binary = cv2.threshold(img, CORNER_BIN_THRESH, 255, cv2.THRESH_BINARY_INV)
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

def get_brightest_point(gray, thresh=THRESH, min_area=MIN_AREA):
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

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

t0 = time.time()
corner_pts = None
sent_corner = False
last_send_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[ymin:ymax, xmin:xmax]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        elapsed = time.time() - t0

        # 前3秒只做角点检测
        if elapsed < 3.0:
            pts = detect_corners(roi)
            if pts is not None:
                corner_pts = pts.copy()
                print(f"角点识别:{corner_pts.tolist()}")
            continue

        # 3秒后每0.2秒检测激光点并串口发送
        now = time.time()
        if now - last_send_time >= 0.2:
            last_send_time = now
            bright_point = get_brightest_point(gray)
            if bright_point:
                x_roi, y_roi = bright_point
                msg1 = f"a{x_roi},{y_roi}z"
                ser.write(msg1.encode('utf-8'))
                print(f"已发送激光点: {msg1}")

                # 只在首次发送激光点时发送角点
                if corner_pts is not None and not sent_corner:
                    pts_flat = corner_pts.flatten()
                    msg2 = 'a' + ','.join(str(int(i)) for i in pts_flat) + 'z'
                    ser.write(msg2.encode('utf-8'))
                    print(f"角点已发送: {msg2}")
                    sent_corner = True

        # 支持Ctrl+C安全退出
except KeyboardInterrupt:
    print("用户终止")

finally:
    cap.release()
    ser.close()
    print("摄像头与串口已关闭")
