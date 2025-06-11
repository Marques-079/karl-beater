# test_predict_on_key_p.py

import cv2
import mss
import numpy as np
import time

from calc import find_ball, find_paddle, mirror_bounce, BBOX

TL_x = BBOX["left"]
sct  = mss.mss()
play_w = BBOX["width"]

def grab_raw():
    return np.array(sct.grab(BBOX))

# Instruction window
cv2.namedWindow("Press P to Sample", cv2.WINDOW_NORMAL)
instr = np.zeros((100,400,3), dtype=np.uint8)
cv2.putText(instr, "Press 'p' to sample trajectory ↑", (10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

print("Focus this window, then press 'p' to sample 1s of ball motion.")

try:
    while True:
        cv2.imshow("Press P to Sample", instr)
        key = cv2.waitKey(50) & 0xFF

        if key == ord('p'):
            # 1) sample 10 positions at 0.1s intervals
            traj = []
            last_raw = None
            for _ in range(10):
                raw = grab_raw()
                last_raw = raw
                bx, by = find_ball(raw)
                traj.append((int(bx), int(by)))
                time.sleep(0.01) #SLEEP HERE

            # 2) detect current paddle and landing from final two samples
            bx0, by0 = traj[-2]
            bx1, by1 = traj[-1]
            # find paddle on the last frame
            _, py = find_paddle(last_raw)
            vx = bx1 - bx0
            vy = by1 - by0

            if vy > 0:
                dt = (py - by1) / vy
                land_x = int(mirror_bounce(bx1 + vx * dt))
            else:
                # if rising, just center
                px, _ = find_paddle(last_raw)
                land_x = int(px)

            # 3) build overlay on the last frame
            disp = last_raw[:, :, :3].copy()
            h, w = disp.shape[:2]

            # draw red trajectory polyline + dots
            pts = np.array(traj, dtype=np.int32)
            cv2.polylines(disp, [pts], False, (0,0,255), 2)
            for x,y in traj:
                cv2.circle(disp, (x,y), 3, (0,0,255), -1)

            # draw green landing line
            cv2.line(disp, (land_x,0), (land_x,h), (0,255,0), 2)

            # annotate
            cv2.putText(disp,
                        f"Land X={land_x}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)

            # 4) show until any key
            cv2.imshow("Prediction", disp)
            cv2.waitKey(0)
            cv2.destroyWindow("Prediction")

        elif key == 27:   # Esc to quit
            break

finally:
    cv2.destroyAllWindows()
