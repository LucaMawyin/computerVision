import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Capturing video
video = cv2.VideoCapture(0)
while True:
    success, frame = video.read()
    if success:
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
