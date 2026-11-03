"""Live Tello camera feed. Press q to quit, s to save a snapshot."""
import time

import cv2

from tello_stream import TelloStream

tello = TelloStream()
tello.start()
print("Waiting for video... (first frame can take a few seconds)")

try:
    while True:
        frame = tello.read()
        if frame is None:
            time.sleep(0.01)
            continue
        cv2.imshow("Tello", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"snap_{int(time.time())}.jpg"
            cv2.imwrite(name, frame)
            print("Saved", name)
finally:
    tello.stop()
    cv2.destroyAllWindows()
