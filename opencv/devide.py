import cv2
import os
import datetime

def main():
    # 可以替换为视频文件路径，例如：'output_20250719_140101.avi'
    video_path = 'output_20250719_151141.avi'  # 使用 0 表示实时摄像头；使用 'your_video.mp4' 表示视频文件

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频或摄像头")
        return

    # 创建保存帧的文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"frames_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"开始保存每一帧到文件夹：{output_dir}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束或错误")
            break

        frame_count += 1
        filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)

        # 可选：显示当前帧
        cv2.imshow("Saving Frames...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中断保存")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"共保存 {frame_count} 帧到：{output_dir}")

if __name__ == "__main__":
    main()
