import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 加载预训练模型（假设你已经训练好了一个 CNN 模型）
model = load_model('digit_recognition_cnn.h5')

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("摄像头打开失败")
    exit()

while True:
    # 捕获每一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        break

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 获取每个数字的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 只选择大于一定大小的区域
        if w > 10 and h > 10:
            # 进行ROI裁剪
            roi = binary[y:y+h, x:x+w]
            # 调整大小为28x28（假设训练时使用的是28x28的输入）
            resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            resized = resized.astype('float32') / 255.0  # 归一化
            resized = np.expand_dims(resized, axis=-1)  # 添加通道维度
            resized = np.expand_dims(resized, axis=0)  # 添加批次维度

            # 使用CNN进行预测
            pred = model.predict(resized)
            predicted_digit = np.argmax(pred, axis=1)[0]

            # 绘制边界框和数字标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(predicted_digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示处理后的图像
    cv2.imshow("Real-time Digit Recognition", frame)

    # 如果按下 'q' 键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
