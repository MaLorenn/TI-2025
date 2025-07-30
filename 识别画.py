import cv2
import numpy as np

# ROI对角线端点
x1, y1 = 496, 284
x2, y2 = 717, 440
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # 只截取并显示ROI
    roi = img[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

    # 用RETR_TREE查找所有轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_output = roi.copy()
    print("ROI轮廓数：", len(contours))
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # 过滤掉小噪声
            if area > 10:
                cv2.drawContours(roi_output, [cnt], -1, (0, 0, 255), 2)
                print(f"ROI轮廓 {i} 面积: {area}，父轮廓编号: {hierarchy[0][i][3]}, 子轮廓编号: {hierarchy[0][i][2]}")

    cv2.imshow('ROI Contours', roi_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
