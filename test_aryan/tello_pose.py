"""Live Tello camera feed with YOLO pose estimation. Press q to quit.

Draws the skeleton and labels a simple gesture per person:
  BOTH ARMS UP / RIGHT ARM UP / LEFT ARM UP / ARMS OUT (T-pose) / none
"""
import time

import cv2
from ultralytics import YOLO

from tello_stream import TelloStream

# COCO keypoint indices
L_SHO, R_SHO, L_WRI, R_WRI = 5, 6, 9, 10
CONF = 0.5


def gesture(kp):
    """kp: (17, 3) array of x, y, conf in image coords (y grows downward)."""
    def ok(i):
        return kp[i][2] > CONF

    if not (ok(L_SHO) and ok(R_SHO)):
        return "none"
    l_up = ok(L_WRI) and kp[L_WRI][1] < kp[L_SHO][1]
    r_up = ok(R_WRI) and kp[R_WRI][1] < kp[R_SHO][1]
    if l_up and r_up:
        return "BOTH ARMS UP"
    if r_up:
        return "RIGHT ARM UP"
    if l_up:
        return "LEFT ARM UP"
    if ok(L_WRI) and ok(R_WRI):
        level = 0.4 * abs(kp[L_SHO][0] - kp[R_SHO][0])
        if (abs(kp[L_WRI][1] - kp[L_SHO][1]) < level
                and abs(kp[R_WRI][1] - kp[R_SHO][1]) < level
                and abs(kp[L_WRI][0] - kp[R_WRI][0]) > 2.5 * abs(kp[L_SHO][0] - kp[R_SHO][0])):
            return "ARMS OUT (T-pose)"
    return "none"


model = YOLO("yolo11n-pose.pt")  # weights sit next to this script (pre-downloaded)
tello = TelloStream()
tello.start()
print("Waiting for video... (first frame can take a few seconds)")

try:
    while True:
        frame = tello.read()
        if frame is None:
            time.sleep(0.01)
            continue

        result = model(frame, verbose=False)[0]
        annotated = result.plot()

        if result.keypoints is not None:
            for i, kp in enumerate(result.keypoints.data.cpu().numpy()):
                label = gesture(kp)
                print(f"\rperson {i}: {label:<20}", end="")
                cv2.putText(annotated, label, (10, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Tello pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    tello.stop()
    cv2.destroyAllWindows()
