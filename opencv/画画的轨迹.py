import cv2
import numpy as np
from skimage.morphology import skeletonize

def resize_for_display(img, max_hw=900):
    h, w = img.shape[:2]
    scale = min(max_hw / h, max_hw / w, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

img = cv2.imread('图7.jpg')
if img is None:
    print('图片读取失败！')
    exit()

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)
_, mask = cv2.threshold(A, 150, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
skeleton = skeletonize(mask_clean // 255)
skeleton = (skeleton * 255).astype(np.uint8)
ys, xs = np.where(skeleton == 255)
points = list(zip(xs, ys))

def is_endpoint(img, x, y):
    roi = img[max(0, y-1):y+2, max(0, x-1):x+2]
    return np.sum(roi==255) == 2

endpoints = [(x, y) for x, y in points if is_endpoint(skeleton, x, y)]
print("骨架端点数：", len(endpoints), endpoints)

# 路径追踪
visited = set()
path = []
def find_next(pt, skel_img, visited):
    x, y = pt
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if dx==0 and dy==0:
                continue
            nx, ny = x+dx, y+dy
            if (nx, ny) in visited:
                continue
            if 0<=nx<skel_img.shape[1] and 0<=ny<skel_img.shape[0]:
                if skel_img[ny, nx]==255:
                    return (nx, ny)
    return None
if endpoints:
    current = endpoints[0]
    path.append(current)
    visited.add(current)
    while True:
        nxt = find_next(current, skeleton, visited)
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        current = nxt
else:
    print("无端点，闭合曲线可用任意点起步")

print(f'轨迹总点数：{len(path)}')

# 固定步长采样
fixed_len = 30  # 每个线段长度
sampled = []
accum_dist = 0
sampled.append(path[0])
for i in range(1, len(path)):
    x0, y0 = path[i-1]
    x1, y1 = path[i]
    d = np.hypot(x1-x0, y1-y0)
    accum_dist += d
    if accum_dist >= fixed_len:
        sampled.append((x1, y1))
        accum_dist = 0
if sampled[-1] != path[-1]:
    sampled.append(path[-1])

print(f'采样端点数：{len(sampled)}')

# 只显示采样端点（红圈）
show = img.copy()
for (x, y) in sampled:
    cv2.circle(show, (x, y), 7, (0, 0, 255), 2)

show_small = resize_for_display(show, 900)
cv2.imshow("Sampled Line Segment Endpoints", show_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
