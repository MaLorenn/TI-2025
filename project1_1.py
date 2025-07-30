import cv2
import numpy as np

def is_circle(contour, threshold=0.8):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > threshold

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败！")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取画面失败")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10 or area > 1000:
                continue
            if is_circle(cnt, threshold=0.8):
                candidates.append(cnt)

        if candidates:
            best = max(candidates, key=cv2.contourArea)
            M = cv2.moments(best)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # 彩色原图绘制
                cv2.drawContours(frame, [best], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Laser Center: ({cx}, {cy})", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 二值图转为3通道方便显示彩色文字
                clean_color = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
                cv2.putText(clean_color, f"({cx}, {cy})", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                clean_color = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
        else:
            clean_color = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Original Frame with Detection", frame)
        cv2.imshow("Gray Frame", gray)
        cv2.imshow("Morphology Open with Coord", clean_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
