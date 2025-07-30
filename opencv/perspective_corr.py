import cv2
import numpy as np

points = []
warped = None
done = False  # 标记是否已完成矫正

def mouse_handler(event, x, y, flags, param):
    global points, image_copy, done
    if event == cv2.EVENT_LBUTTONDOWN and not done:
        if len(points) < 4:
            points.append([x, y])
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Input", image_copy)

image = cv2.imread("0.jpg")
if image is None:
    print("无法加载图像，请确认路径")
    exit()

image_copy = image.copy()
cv2.imshow("Input", image_copy)
cv2.setMouseCallback("Input", mouse_handler)

print("请点击图像中 A4 纸的四个角。按 'r' 重置，'q' 退出。")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        points.clear()
        done = False
        image_copy = image.copy()
        cv2.imshow("Input", image_copy)
        if warped is not None:
            cv2.destroyWindow("Warped A4")
        print("已重置，请重新点击四个角。")

    elif key == ord('q'):
        break

    # 四点已选且未处理
    if len(points) == 4 and not done:
        pts_src = np.array(points, dtype="float32")
        width, height = 595, 842

        pts_dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        cv2.imshow("Warped A4", warped)
        cv2.imwrite("a4_corrected.jpg", warped)
        print("矫正完成，已保存为 a4_corrected.jpg")

        done = True  # 防止重复计算

cv2.destroyAllWindows()
