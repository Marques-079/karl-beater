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
            # 1) Sample 10 positions very quickly
            traj = []
            last_raw = None
            for _ in range(10):
                raw = grab_raw()
                last_raw = raw
                bx, by = find_ball(raw)
                traj.append((bx, by))
                time.sleep(0.005)  # faster sampling

            # 2) Grab paddle Y from the final frame
            _, paddle_y = find_paddle(last_raw)

            # 3) Build the display frame
            disp = last_raw[:, :, :3].copy()
            h, w = disp.shape[:2]

            # Draw the red trajectory
            pts = np.array(traj, dtype=np.int32)
            cv2.polylines(disp, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            for x, y in traj:
                cv2.circle(disp, (int(x), int(y)), 3, (0, 0, 255), -1)

            # 4) Simulate the rebound path
            # Start from the last sampled point and its velocity
            x0, y0 = traj[-2]
            x1, y1 = traj[-1]
            vx = x1 - x0
            vy = y1 - y0

            # Neon‐green in BGR
            neon = (57, 255, 20)

            # Collect segments as list of point‐pairs
            segments = []
            pos = np.array([x1, y1], dtype=float)
            vel = np.array([vx, vy], dtype=float)

            # Keep simulating until we hit paddle_y
            while True:
                # time to vertical paddle line
                if vel[1] <= 0:
                    # moving up or zero, break out
                    break
                t_to_paddle = (paddle_y - pos[1]) / vel[1]

                # time to hit left/right walls
                if vel[0] > 0:
                    t_wall = (w - pos[0]) / vel[0]
                    wall_x = w
                else:
                    t_wall = (0 - pos[0]) / vel[0]
                    wall_x = 0

                if 0 <= t_to_paddle <= t_wall:
                    # we hit paddle before wall
                    end_pos = pos + vel * t_to_paddle
                    segments.append((tuple(pos.astype(int)), tuple(end_pos.astype(int))))
                    break
                else:
                    # we bounce off a wall first
                    if t_wall < 0:
                        # would hit behind us—abort
                        break
                    hit_pos = pos + vel * t_wall
                    segments.append((tuple(pos.astype(int)), tuple(hit_pos.astype(int))))
                    # reflect horizontal velocity
                    vel[0] *= -1
                    pos = hit_pos  # continue from the bounce point

            # 5) Draw neon‐green rebound segments
            for (sx, sy), (ex, ey) in segments:
                cv2.line(disp, (sx, sy), (ex, ey), neon, 2, lineType=cv2.LINE_AA)
                cv2.circle(disp, (ex, ey), 4, neon, -1)  # marker at each segment end

            # 6) Show it
            cv2.putText(disp, f"Land X={int(segments[-1][1][0])}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, neon, 2)
            cv2.imshow("Prediction", disp)
            cv2.waitKey(0)
            cv2.destroyWindow("Prediction")


finally:
    cv2.destroyAllWindows()
