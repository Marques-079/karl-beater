# test_predict_visual.py

import cv2
import mss
import numpy as np
import time

# bring in your existing detector + maths
from calc import find_ball, find_paddle, mirror_bounce, BBOX

#----------- 1) Setup & calibration -----------

TL_x = BBOX["left"]
sct  = mss.mss()

def grab_raw():
    """Grab the calibrated game window as BGRA."""
    return np.array(sct.grab(BBOX))

# Little instruction panel
cv2.namedWindow("Press P to Predict", cv2.WINDOW_NORMAL)
blank = np.zeros((100, 400, 3), dtype=np.uint8)
cv2.putText(
    blank,
    "Press 'p' to predict ↑",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
)

print("→ Focus this window, then press 'p' during play to see 5-step overlay.")
print("→ Press ESC or Ctrl+C to quit.")

#----------- 2) Main loop -----------

try:
    while True:
        # show instructions
        cv2.imshow("Press P to Predict", blank)
        key = cv2.waitKey(50) & 0xFF

        if key == ord("p"):
            # 2.1) Grab 6 frames (5 intervals)
            raws = []
            for i in range(6):
                raws.append(grab_raw())
                if i < 5:
                    time.sleep(1/30)

            # 2.2) Compute 5 predictions
            preds = []
            # we'll need the last paddle-y for dot placement
            last_py = None

            for i in range(1, 6):
                prev = raws[i - 1]
                cur  = raws[i]

                bx1, by1 = find_ball(prev)
                bx2, by2 = find_ball(cur)
                px2, py2 = find_paddle(cur)
                last_py = py2

                vx = bx2 - bx1
                vy = by2 - by1

                if vy <= 0:
                    land_x = px2
                else:
                    dt = (py2 - by2) / vy
                    land_x = int(mirror_bounce(bx2 + vx * dt))

                preds.append((bx2, by2, land_x, vx, vy))

            # 2.3) Overlay all predictions on the 6th frame
            disp = raws[-1][:, :, :3].copy()
            h, w = disp.shape[:2]

            for idx, (bx, by, land_x, vx, vy) in enumerate(preds):
                # color ramp: blue → green
                t     = idx / 4
                color = (int((1 - t) * 255), int(t * 255), 0)

                # instantaneous velocity arrow
                start = (int(bx), int(by))
                end   = (
                    int(bx + vx * 10),
                    int(by + vy * 10),
                )
                cv2.arrowedLine(disp, start, end, color, 1, tipLength=0.3)

                # predicted landing vertical line
                cv2.line(
                    disp,
                    (land_x, 0),
                    (land_x, h),
                    color,
                    1,
                    lineType=cv2.LINE_AA,
                )

                # dot at the paddle level
                cv2.circle(disp, (int(land_x), int(last_py)), 4, color, -1)


            # 2.4) Annotate & display
            cv2.putText(
                disp,
                "5-step predictions",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Prediction", disp)
            cv2.waitKey(0)
            cv2.destroyWindow("Prediction")

            # cleanup
            del raws, preds, disp

        elif key == 27:  # ESC
            break

finally:
    cv2.destroyAllWindows()
