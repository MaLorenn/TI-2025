import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def sort_pts(pts):
    # 左上、右上、右下、左下排序
    pts = sorted(pts, key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return np.array(top + bottom)

while True:
    ret, img_raw = cap.read()
    if not ret:
        break

    # 转灰度
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # 双边滤波
    img = cv2.bilateralFilter(img, 9, 10, 10)
    # 闭运算
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 二值化
    _, binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
    # 查找轮廓（重点改为RETR_TREE！）
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    approx_outer = None
    approx_inner = None

    if hierarchy is not None:
        # hierarchy[0][i][3] == -1 是外轮廓，其余为内轮廓
        # 按面积排序，从大到小取出两个四边形轮廓，分别归类
        quads = []
        for i, contour in enumerate(contours):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                quads.append((area, approx, hierarchy[0][i][3]))
        quads = sorted(quads, key=lambda x: -x[0])  # 面积降序排列

        # 归类：父轮廓-1为外，其他为内
        for area, approx, parent in quads:
            if parent == -1 and approx_outer is None:
                approx_outer = approx.reshape(-1, 2)
            elif parent != -1 and approx_inner is None:
                approx_inner = approx.reshape(-1, 2)
            if approx_outer is not None and approx_inner is not None:
                break

        # 绘制与计算
        if approx_outer is not None:
            pts0 = sort_pts(approx_outer)
            cv2.drawContours(img_raw, [pts0], 0, (0, 255, 0), 2)
            for x, y in pts0:
                cv2.circle(img_raw, (x, y), 5, (255, 0, 0), 2)
        if approx_inner is not None:
            pts1 = sort_pts(approx_inner)
            cv2.drawContours(img_raw, [pts1], 0, (0, 200, 255), 2)
            for x, y in pts1:
                cv2.circle(img_raw, (x, y), 5, (0, 0, 255), 2)
        if approx_outer is not None and approx_inner is not None:
            pts0 = sort_pts(approx_outer)
            pts1 = sort_pts(approx_inner)
            for i in range(4):
                center = ((pts0[i] + pts1[i]) / 2).astype(int)
                cv2.circle(img_raw, tuple(center), 7, (0, 0, 255), -1)
                cv2.putText(img_raw, f"{i+1}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Rect Corner Detection', img_raw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
