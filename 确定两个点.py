import cv2

points = []
roi_defined = False
xmin = xmax = ymin = ymax = 0

def mouse_callback(event, x, y, flags, param):
    global points, roi_defined, xmin, xmax, ymin, ymax
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"第 {len(points)} 个点: {x}, {y}")
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            roi_defined = True
            points = []

cap = cv2.VideoCapture(2)  # 可改为2

# 设置高分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_disp = frame.copy()
    # 显示已选点和矩形
    if len(points) == 1:
        cv2.circle(frame_disp, points[0], 5, (0, 0, 255), -1)
    if roi_defined:
        cv2.rectangle(frame_disp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('Image', frame_disp)

    # 如果已经选定ROI，显示ROI实时小窗口
    if roi_defined and xmax > xmin and ymax > ymin:
        roi = frame[ymin:ymax, xmin:xmax]
        if roi.size > 0:
            cv2.imshow('ROI Live', roi)
    else:
        # 如果未选定或区域无效，关闭小窗口
        cv2.destroyWindow('ROI Live')

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
