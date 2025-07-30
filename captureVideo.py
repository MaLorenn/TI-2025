# -*- coding: utf-8 -*-
import cv2
import time

# 打开默认摄像头（通常是 /dev/video0 或 Windows 上的第一个摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

frame_count = 0
start_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # 显示帧率
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
