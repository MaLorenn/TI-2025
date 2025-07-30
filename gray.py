# -*- coding: utf-8 -*-
import cv2

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

clicked_point = None
gray_value = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point, gray_value
    if event == cv2.EVENT_LBUTTONDOWN and param is not None:
        clicked_point = (x, y)
        gray_value = param[y, x]
        print(f"点击像素点({x}, {y})，灰度值: {gray_value}")

cv2.namedWindow("Camera")
# 注意：mouse_callback 的 param 需要在主循环里每帧重新设置
cv2.setMouseCallback("Camera", mouse_callback, None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 给鼠标回调实时传递gray_frame参数
    cv2.setMouseCallback("Camera", mouse_callback, gray_frame)

    # 可视化点击点和灰度
    frame_disp = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    if clicked_point is not None and gray_value is not None:
        cv2.circle(frame_disp, clicked_point, 7, (0, 0, 255), 2)
        cv2.putText(frame_disp, f'{gray_value}', (clicked_point[0]+10, clicked_point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Camera", frame_disp)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
