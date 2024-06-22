import cv2
from libs.video import camera

stream = camera()
stream.start()

while True:
    frame = stream.read()
    cv2.imshow("camera",frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()