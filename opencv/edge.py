import cv2
import numpy as np

# 1. 读取图像
image = cv2.imread('input.jpg')  # 替换为你的图片路径
if image is None:
    print("图像加载失败，请检查文件路径！")
    exit()

# 2. 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 高斯模糊（去噪，效果更稳）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. 使用 Canny 算法做边缘检测
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# 5. 查找轮廓（边缘检测图中）
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. 在原图上绘制轮廓
contour_img = image.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# 显示结果
cv2.imshow('原图', image)
cv2.imshow('Canny边缘', edges)
cv2.imshow('轮廓图', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
