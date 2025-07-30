import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import deque

# 固定两个点
pt1 = (729, 288)
pt2 = (505, 441)

def nothing(x):
    pass

cap = cv2.VideoCapture(2)  # 摄像头编号请根据实际调整
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("thresh")
cv2.createTrackbar("thresh", "thresh", 100, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败")
        break

    x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    y_min, y_max = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
    roi = frame[y_min:y_max, x_min:x_max]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh_val = cv2.getTrackbarPos("thresh", "thresh")
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # 骨架化
    skeleton = skeletonize(binary > 0)
    skeleton = (skeleton * 255).astype(np.uint8)

    # 骨架采样
    coords = np.column_stack(np.where(skeleton == 255))  # [[y, x], ...]
    sampled_on_img = []
    if len(coords) > 0:
        visited = np.zeros_like(skeleton, dtype=bool)

        def distance(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        start = tuple(coords[0])  # (y, x)
        sampled = [start]
        queue = deque([start])
        visited[start[0], start[1]] = True
        last_sample = start

        while queue:
            pt = queue.popleft()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = pt[0] + dy, pt[1] + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx] and not visited[ny, nx]:
                            queue.append((ny, nx))
                            visited[ny, nx] = True
                            if distance((ny, nx), last_sample) >= 10:
                                sampled.append((ny, nx))
                                last_sample = (ny, nx)
        # 采样点映射回frame坐标（记得是x + x_min, y + y_min）
        sampled_on_img = [(x + x_min, y + y_min) for (y, x) in sampled]

    # 可视化
    show = frame.copy()
    cv2.rectangle(show, pt1, pt2, (0, 255, 0), 2)
    for pt in sampled_on_img:
        cv2.circle(show, pt, 2, (0, 0, 255), -1)
    for i in range(1, len(sampled_on_img)):
        cv2.line(show, sampled_on_img[i - 1], sampled_on_img[i], (255, 0, 0), 1)

    cv2.imshow("Camera Route", show)
    cv2.imshow("thresh", binary)

    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.imwrite("saved_binary.png", binary)
        cv2.imwrite("saved_route.png", show)
        print("当前帧已保存")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
