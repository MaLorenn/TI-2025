import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 转为 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色的 HSV 范围（两个区间）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 创建掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 计算掩码中非零（红色）像素的数量
    red_area = cv2.countNonZero(red_mask)

    # 判断是否大于阈值
    if red_area > 2000:
        print("条件一")

    # 显示原始画面和掩码
    cv2.imshow("Frame", frame)
    cv2.imshow("Red Mask", red_mask)

    # 按下 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
