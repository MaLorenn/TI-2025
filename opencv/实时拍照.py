import cv2
import time

cap = cv2.VideoCapture(2)  # 摄像头编号按实际情况修改
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYU'))

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    if key == ord(' '):  # 空格键拍照
        filename = f"photo_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"已保存：{filename}")

    if key == ord('q'):  # q键退出
        break

cap.release()
cv2.destroyAllWindows()
