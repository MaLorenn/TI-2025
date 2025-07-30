import cv2
import numpy as np

def configure_camera(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def get_all_contour_centers(mask, min_area=1, max_area=120):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
    return centers

def get_largest_contour_center(mask, min_area=35, max_area=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    if not valid:
        return None
    best = max(valid, key=cv2.contourArea)
    M = cv2.moments(best)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("摄像头打开失败")
        return

    configure_camera(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        x1 = w // 3
        x2 = 2 * w // 3
        y1 = h // 3
        y2 = 2 * h // 3
        roi = frame[y1:y2, x1:x2]  # 九宫格中心区域

        # 预处理
        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # 亮点提取
        _, bright_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

        # 红色区域提取
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 找中心点
        red_center = get_largest_contour_center(red_mask)
        bright_centers = get_all_contour_centers(bright_mask)

        if red_center and bright_centers:
            closest = min(bright_centers, key=lambda p: euclidean_dist(p, red_center))

            # 还原坐标到原图
            red_center_full = (red_center[0] + x1, red_center[1] + y1)
            closest_full = (closest[0] + x1, closest[1] + y1)

            # 显示圆和文字
            cv2.circle(frame, red_center_full, 7, (0, 0, 255), 2)
            cv2.circle(frame, closest_full, 7, (0, 255, 0), 2)
            cv2.putText(frame, f"Red Laser Center: {closest_full}",
                        (closest_full[0] + 10, closest_full[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示九宫格边界线
        for i in range(1, 3):
            cv2.line(frame, (0, i * h // 3), (w, i * h // 3), (100, 255, 255), 1)
            cv2.line(frame, (i * w // 3, 0), (i * w // 3, h), (100, 255, 255), 1)

        # 显示画面
        cv2.imshow("Frame", frame)
        cv2.imshow("Bright Mask", bright_mask)
        cv2.imshow("Red Mask", red_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
