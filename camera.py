import cv2
import subprocess
import json
import os
import time  # 用于 FPS 计算

device = "/dev/video0"
settings_file = "camera_settings.json"

def set_v4l2_ctrl(name, value):
    subprocess.call(f"v4l2-ctl -d {device} --set-ctrl={name}={value}", shell=True)

def get_trackbar(name):
    return cv2.getTrackbarPos(name, 'Camera Controls')

# 初始化摄像头参数
def apply_saved_settings():
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        for key, value in settings.items():
            set_v4l2_ctrl(key, value)
        print("已加载保存的参数设置")
    else:
        print("未找到参数文件，使用默认设置")

# 关闭自动功能
set_v4l2_ctrl("auto_exposure", 1)                # 曝光手动
set_v4l2_ctrl("white_balance_automatic", 0)      # 关闭自动白平衡

# 应用之前保存的设置（如果有）
apply_saved_settings()

# 打开摄像头
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 创建窗口
cv2.namedWindow('Camera Controls')

# 创建滑动条
cv2.createTrackbar('Brightness', 'Camera Controls', 41, 128,
                   lambda v: set_v4l2_ctrl('brightness', v - 64))
cv2.createTrackbar('Exposure', 'Camera Controls', 429, 10000,
                   lambda v: set_v4l2_ctrl('exposure_time_absolute', v))
cv2.createTrackbar('Color Temp', 'Camera Controls', 4290, 6500,
                   lambda v: set_v4l2_ctrl('white_balance_temperature', v))
cv2.createTrackbar('Gain', 'Camera Controls', 59, 128,
                   lambda v: set_v4l2_ctrl('gain', v))
cv2.createTrackbar('Contrast', 'Camera Controls', 32, 64,
                   lambda v: set_v4l2_ctrl('contrast', v))  # ★ 新增对比度

# 初始化帧率计算
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面")
        break

    # 帧率计算
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    else:
        fps = None

    # 显示帧率（如果已计算）
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Camera Controls', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 保存当前参数
        settings = {
            "brightness": get_trackbar("Brightness") - 64,
            "exposure_time_absolute": get_trackbar("Exposure"),
            "white_balance_temperature": get_trackbar("Color Temp"),
            "gain": get_trackbar("Gain"),
            "contrast": get_trackbar("Contrast")
        }
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        print("已保存当前设置到", settings_file)
        break

cap.release()
cv2.destroyAllWindows()
