import cv2
import datetime

def configure_camera(cap):
    # 设置 MJPG 格式以提升帧率
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # 设置帧率
    cap.set(cv2.CAP_PROP_FPS, 30)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    configure_camera(cap)

    # 获取实际帧率（有些摄像头无法精确控制）
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 生成时间戳文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.avi"

    # 使用 VideoWriter 保存录像（MJPG 编码）
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"开始录制视频，保存为：{output_filename}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        out.write(frame)  # 保存一帧
        cv2.imshow("Camera Preview", frame)

        # 按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频录制结束，文件保存在：{output_filename}")

if __name__ == "__main__":
    main()
