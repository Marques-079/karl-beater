import cv2
import mss
import numpy as np
import time
import pyautogui
import math
from calc_final import find_ball, find_paddle, mirror_bounce, BBOX

# ── Calibration & constants ────────────────────────────────────────────────
TL, TT = BBOX['left'], BBOX['top']
PLAY_W, PLAY_H = BBOX['width'], BBOX['height']

SAMPLES_INIT = 15
BALL_RADIUS = 14.5
LEFT_BOUND = BALL_RADIUS
RIGHT_BOUND = PLAY_W - BALL_RADIUS
GROUND_OFFSET = 40
Y_MIN = PLAY_H * 0.15
Y_MAX = PLAY_H * 0.35

# Custom desktop scaling parameters
# relative frame width=565 maps to desktop span=318, offset=14
FRAME_W = 565.0
DESKTOP_SPAN = 318.0
DESKTOP_OFFSET = 14.0

sct = mss.mss()
cv2.namedWindow("Trigger", cv2.WINDOW_NORMAL)

# Prebind for speed
grab = lambda: np.array(sct.grab(BBOX))
fb = find_ball
fp = find_paddle
m_bounce = mirror_bounce
moveTo = pyautogui.moveTo
dragTo = pyautogui.dragTo

contacts = 0
run_detection = False
screen_y = None

# Contact sampler
def get_samples_count():
    global contacts
    contacts += 1
    return 6 if contacts > SAMPLES_INIT else SAMPLES_INIT

try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

        # ── On first 'p', grab paddle and begin drag
        if not run_detection and key == ord('p'):
            moveTo(TL + 10, TT + 10, duration=0)
            time.sleep(0.2)
            frame = grab()
            px, py = fp(frame)
            if px is None:
                continue
            screen_x = TL + int(px) - 150
            screen_y = TT + int(py) - 720
            moveTo(screen_x, screen_y, duration=0)
            run_detection = True

        if not run_detection:
            continue

        # ── Wait for trigger: ball moving down into band
        prev_y = None
        while True:
            raw0 = grab()
            bx0, by0 = fb(raw0)
            if bx0 is None:
                continue
            if prev_y is not None and by0 > prev_y and Y_MIN <= by0 <= Y_MAX:
                break
            prev_y = by0

        # ── Sample trajectory
        traj = []
        last = None
        for _ in range(get_samples_count()):
            raw = grab(); last = raw
            bx, by = fb(raw)
            if traj and len(traj) > 8 and (bx < 30 or bx > 490):
                break
            traj.append((bx, by))

        # ── Filter straight majority (±6°)
        N = len(traj)
        if N >= 2:
            angles = [math.atan2(traj[i][1]-traj[i-1][1], traj[i][0]-traj[i-1][0]) for i in range(1, N)]
            tol = math.radians(6)
            def circ_diff(a,b): return abs(((a-b)+math.pi)%(2*math.pi)-math.pi)
            best_angle = max(angles, key=lambda cand: sum(1 for a in angles if circ_diff(a, cand) <= tol))
            mask = [circ_diff(a, best_angle) <= tol for a in angles]
            keep = {i for i, ok in enumerate(mask) if ok} | {i+1 for i, ok in enumerate(mask) if ok}
            if sum(mask) > len(angles)/2 and len(keep) >= 2:
                traj = [traj[i] for i in sorted(keep)]

        # ── Bounce prediction
        if len(traj) < 2:
            continue
        x0, y0 = traj[-2]
        x1, y1 = traj[-1]
        vel = np.array([x1-x0, y1-y0], float)
        pos = np.array([x1, y1], float)
        paddle_y = fp(last)[1]
        ground_y = paddle_y - GROUND_OFFSET
        w_img = last.shape[1]
        segs = []
        while vel[1] > 0:
            t_s = (ground_y - pos[1]) / vel[1]
            t_w = ((w_img if vel[0]>0 else 0) - pos[0]) / vel[0]
            if 0 <= t_s <= t_w:
                segs.append((pos.copy(), pos + vel*t_s)); break
            if t_w < 0: break
            hit = pos + vel*t_w
            segs.append((pos.copy(), hit.copy()))
            vel[0] *= -1; pos = hit.copy()
        local_x = segs[-1][1][0] if segs else m_bounce(x1)

        # ── Custom scaling to desktop coords
        desktop_x = (local_x / FRAME_W) * DESKTOP_SPAN + DESKTOP_OFFSET
        dragTo(desktop_x, screen_y, duration=0.02, button='left')

except KeyboardInterrupt:
    pass
finally:
    if run_detection:
        pyautogui.mouseUp()
    cv2.destroyAllWindows()
