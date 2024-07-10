import cv2
import numpy as np

def detect_pitch_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    return lines

# Example usage
frame = cv2.imread('your_image.png')
lines = detect_pitch_lines(frame)

# Draw detected lines
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Detected Lines', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
