import cv2
import numpy as np
from skimage.morphology import skeletonize

# 读取并预处理图片
img = cv2.imread('图6.jpg')
if img is None:
    print('图片读取失败！')
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # 紫线偏浅色，取反黑底白线
skeleton = skeletonize(binary // 255)
skeleton = (skeleton * 255).astype(np.uint8)

# 找所有骨架点
ys, xs = np.where(skeleton == 255)
coords = set(zip(xs, ys))

# --- 游走全骨架并采样 ---
def fixed_step_traversal(coords, skeleton, step=20):
    if not coords:
        return []

    # 从一个点出发
    current = next(iter(coords))
    path = [current]
    visited = set([current])
    sampled = [current]
    last_sample = current
    accum_dist = 0

    while len(visited) < len(coords):
        x, y = current
        neighbors = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x+dx, y+dy
                if (nx, ny) in coords and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
        if not neighbors:
            # 如果断线/死胡同，跳到还未访问的任意点（处理闭合和交叉）
            left = coords - visited
            if not left:
                break
            current = next(iter(left))
            path.append(current)
            visited.add(current)
            accum_dist = 0  # 断线处不采样
            continue
        # 选距离上一个点最近的邻居（贪心，不会乱窜）
        nxt = min(neighbors, key=lambda p: np.hypot(p[0]-x, p[1]-y))
        path.append(nxt)
        visited.add(nxt)
        accum_dist += np.hypot(nxt[0]-last_sample[0], nxt[1]-last_sample[1])
        if accum_dist >= step:
            sampled.append(nxt)
            last_sample = nxt
            accum_dist = 0
        current = nxt
    if sampled[-1] != path[-1]:
        sampled.append(path[-1])
    return sampled

sampled_points = fixed_step_traversal(coords, skeleton, step=20)

# 可视化采样点和采样路径
show = img.copy()
for (x, y) in sampled_points:
    cv2.circle(show, (x, y), 6, (0, 0, 255), 2)
for i in range(1, len(sampled_points)):
    cv2.line(show, sampled_points[i-1], sampled_points[i], (255, 0, 0), 1)

show = cv2.resize(show, (show.shape[1]*2, show.shape[0]*2))  # 放大便于观察
cv2.imshow("Sampled 8-shape", show)
cv2.waitKey(0)
cv2.destroyAllWindows()
