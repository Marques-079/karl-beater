import cv2
import mss
import numpy as np
import time

from calc_final import find_ball, find_paddle, mirror_bounce, BBOX

# ── Calibration & constants ────────────────────────────────────────────────
play_w, play_h = BBOX["width"], BBOX["height"]

SAMPLES = 10            # samples for trajectory fit
SAMPLE_DT = 0.00      # sec between samples

BALL_RADIUS = 14.8
LEFT_BOUND = BALL_RADIUS
RIGHT_BOUND = play_w - BALL_RADIUS

ground_offset = 40      # slider‐top line above paddle

# 65 %–75 % vertical band where we want to catch the ball moving down
Y_MIN = play_h * 0.65
Y_MAX = play_h * 0.75

sct = mss.mss()


def grab_raw():
    """Grab the calibrated play area as a BGRA image."""
    return np.array(sct.grab(BBOX))


def mirror_bounce_adj(x):
    """Analytic reflection of x between LEFT_BOUND and RIGHT_BOUND."""
    span = RIGHT_BOUND - LEFT_BOUND
    pos = x - LEFT_BOUND
    m = pos % (2 * span)
    xr = m if m <= span else (2 * span - m)
    return xr + LEFT_BOUND


cv2.namedWindow("Trigger", cv2.WINDOW_NORMAL)

# ── Main loop ──────────────────────────────────────────────────────────────
run_detection = False  # becomes True after first 'P' press

try:
    while True:
        key = cv2.waitKey(1) & 0xFF  # small delay keeps UI responsive
        if key == 27:                # Esc exits programme
            break
        if key == ord("p"):
            run_detection = True     # arm continuous detection

        if not run_detection:
            continue                 # waiting for initial 'P'

        # ── 1) Wait for ball entering the band while moving down ───────────
        prev_y = None
        while True:
            raw0 = grab_raw()
            bx0, by0 = find_ball(raw0)
            if bx0 is None:
                continue
            if prev_y is not None and (by0 > prev_y) and (Y_MIN <= by0 <= Y_MAX):
                break                # ball is in band & descending
            prev_y = by0
            time.sleep(1 / 60)

        # ── 2) Sample short trajectory ─────────────────────────────────────
        traj = []
        last = None
        for _ in range(SAMPLES):
            raw = grab_raw()
            last = raw
            bx, by = find_ball(raw)
            traj.append((bx, by))
            time.sleep(SAMPLE_DT)

        # ── 3) Predict landing x at slider‐top line ────────────────────────
        _, paddle_y = find_paddle(last)
        ground_y = paddle_y - ground_offset

        x0, y0 = traj[-2]
        x1, y1 = traj[-1]
        vel = np.array([x1 - x0, y1 - y0], float)
        pos = np.array([x1, y1], float)
        w = last.shape[1]
        segs = []

        while vel[1] > 0:  # while still moving downward
            t_s = (ground_y - pos[1]) / vel[1]
            t_w = ((w if vel[0] > 0 else 0) - pos[0]) / vel[0]

            if 0 <= t_s <= t_w:
                segs.append((pos.copy(), pos + vel * t_s))
                break
            if t_w < 0:
                break

            hit = pos + vel * t_w
            segs.append((pos.copy(), hit.copy()))
            vel[0] *= -1              # horizontal velocity flips on wall
            pos = hit

        final_x = int(segs[-1][1][0]) if segs else int(mirror_bounce_adj(x1))

        # ── 4) Print landing coordinates ───────────────────────────────────
        print(f"Magenta dot at X={final_x}, Y={int(ground_y)}")

        # loop continues automatically → waits for next descent

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()


