import cv2

def nothing(x):
    pass  # 滑轨回调函数占位，不用做事

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

cv2.namedWindow('Original')
cv2.namedWindow('Gray')
cv2.namedWindow('Binary Threshold')

# 创建滑轨，范围0~255，初始值127
cv2.createTrackbar('Threshold', 'Binary Threshold', 127, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 获取当前滑轨阈值
    thresh_val = cv2.getTrackbarPos('Threshold', 'Binary Threshold')

    # 二值化处理
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # 显示二值图像
    cv2.imshow('Original', frame)
    cv2.imshow('Gray', gray)
    cv2.imshow('Binary Threshold', binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
