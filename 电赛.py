import cv2
import numpy as np

def order_points(pts):
    # 角点排序为：左上、右上、右下、左下
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

cap = cv2.VideoCapture(0)  # 选择摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 转灰度+高斯模糊（降噪）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # 2. 二值化（低阈值选黑色区域）
    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)

    # 3. 找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 4. 面积过滤
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        # 5. 多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 6. 只保留四边形
        if len(approx) == 4:
            # 透视角点排序
            pts = approx.reshape(4, 2)
            rect = order_points(pts)

            # 画框
            cv2.polylines(frame, [np.int32(rect)], True, (0, 255, 0), 2)

            # 计算中心
            M = cv2.moments(rect)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # 或者直接四点均值
                cx, cy = np.mean(rect, axis=0).astype(int)

            # 标注中心点
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
