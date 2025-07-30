import cv2

# 固定两个点作为ROI区域
x1, y1 = 238, 196
x2, y2 = 385, 295
xmin, xmax = min(x1, x2), max(x1, x2)
ymin, ymax = min(y1, y2), max(y1, y2)

cap = cv2.VideoCapture(2)  # 如需用2改为2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 只显示ROI窗口
    roi = frame[ymin:ymax, xmin:xmax]
    if roi.size > 0:
        cv2.imshow('ROI Live', roi)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
