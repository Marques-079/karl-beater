# trigger_predict.py

import cv2
import mss
import numpy as np
import time

from calc_final import find_ball, find_paddle, mirror_bounce, BBOX

#── 1) Calibration & constants ──────────────────────────────────────────────
TL_x, TL_y = BBOX["left"], BBOX["top"]
play_w, play_h = BBOX["width"], BBOX["height"]

# sampling
SAMPLES   = 10
SAMPLE_DT = 0.0025  # sec

# ball radius → adjusted bounce
BALL_RADIUS = 14.8
LEFT_BOUND  = BALL_RADIUS
RIGHT_BOUND = play_w - BALL_RADIUS

# ground line: 40px above paddle
GROUND_OFF = 40

# colors (BGR)
RED     = (0,   0, 255)
NEON    = (57,255, 20)
BLACK   = (0,   0,   0)
MAGENTA = (255,  0,255)

# vertical band: 65%–75% of the play area height
Y_MIN = play_h * 0.65
Y_MAX = play_h * 0.75

sct = mss.mss()

def grab_raw():
    """Grab calibrated window as BGRA."""
    return np.array(sct.grab(BBOX))

def mirror_bounce_adj(x):
    """Reflect around [LEFT_BOUND, RIGHT_BOUND]."""
    span = RIGHT_BOUND - LEFT_BOUND
    pos  = x - LEFT_BOUND
    m    = pos % (2*span)
    xr   = m if m <= span else (2*span - m)
    return xr + LEFT_BOUND

#── 2) Instruction window ───────────────────────────────────────────────────
cv2.namedWindow("Press P to Trigger", cv2.WINDOW_NORMAL)
instr = np.zeros((100,400,3), dtype=np.uint8)
cv2.putText(instr,
            "Press 'P' to start ↓-band detection",
            (10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

print("Focus this window and press 'P' at the right moment.")
print("Esc to quit.")

try:
    while True:
        cv2.imshow("Press P to Trigger", instr)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # Esc to exit
            break

        if key == ord('p'):
            # ─── WAIT FOR BALL IN 65%–75% BAND MOVING DOWN ───────────────
            print(f"⏳ Waiting for ball in {65:.0f}%–{75:.0f}% band moving ↓ …")
            prev_y = None

            while True:
                raw0 = grab_raw()
                bx0, by0 = find_ball(raw0)
                if bx0 is None:
                    continue  # could not find ball; retry

                if prev_y is not None:
                    downward = (by0 > prev_y)
                    in_band  = (Y_MIN <= by0 <= Y_MAX)
                    if downward and in_band:
                        print(f"✅ Triggered at x={bx0:.1f}, y={by0:.1f}")
                        break

                prev_y = by0
                time.sleep(1/60)

            # ─── SAMPLE TRAJECTORY ────────────────────────────────────────
            traj = []
            last = None
            for _ in range(SAMPLES):
                raw = grab_raw()
                last = raw
                bx, by = find_ball(raw)
                traj.append((bx, by))
                time.sleep(SAMPLE_DT)

            # ─── PREDICTION & DRAW ────────────────────────────────────────
            _, paddle_y = find_paddle(last)
            ground_y    = paddle_y - GROUND_OFF

            disp = last[:, :, :3].copy()
            h, w = disp.shape[:2]

            # draw past trajectory (red)
            pts = np.array(traj, dtype=np.int32)
            cv2.polylines(disp, [pts], False, RED, 2)
            for x, y in traj:
                cv2.circle(disp, (int(x), int(y)), 3, RED, -1)

            # compute and draw bounce prediction (neon),
            # now targeting ground_y instead of paddle_y
            x0, y0 = traj[-2]
            x1, y1 = traj[-1]
            vel = np.array([x1 - x0, y1 - y0], float)
            pos = np.array([x1, y1], float)
            segs = []

            while vel[1] > 0:
                # time to hit the slider-top (ground_y)
                t_s = (ground_y - pos[1]) / vel[1]
                if vel[0] > 0:
                    t_w = (w - pos[0]) / vel[0]
                else:
                    t_w = (0 - pos[0]) / vel[0]

                # if the slider‐hit comes before any wall‐hit, we're done
                if 0 <= t_s <= t_w:
                    end = pos + vel * t_s
                    segs.append((pos.copy(), end.copy()))
                    break

                # no valid slider‐hit, and if wall‐hit is negative → abort
                if t_w < 0:
                    break

                # bounce off wall
                hit = pos + vel * t_w
                segs.append((pos.copy(), hit.copy()))
                vel[0] *= -1
                pos = hit

            # draw neon‐prediction segments
            for a, b in segs:
                cv2.line(
                    disp,
                    tuple(map(int, a)),
                    tuple(map(int, b)),
                    NEON, 2, cv2.LINE_AA
                )

            # draw slider‐top line + magenta landing dot
            cv2.line(disp, (0, int(ground_y)), (w, int(ground_y)), BLACK, 2)

            if segs:
                final_x = int(segs[-1][1][0])
            else:
                # fallback to analytic mirror_bounce targeting ground_y
                final_x = int(mirror_bounce((x1, y1), ground_y))

            cv2.circle(disp, (final_x, int(ground_y)), 6, MAGENTA, -1)

            cv2.putText(disp,
                        f"Land X={final_x}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, NEON, 2)

            cv2.imshow("Prediction", disp)
            cv2.waitKey(0)
            cv2.destroyWindow("Prediction")

    cv2.destroyAllWindows()

except KeyboardInterrupt:
    cv2.destroyAllWindows()
