# trigger_predict.py

import time
import mss
import numpy as np
import cv2

from calc2 import find_ball, find_paddle, mirror_bounce, BBOX

#── 1) Calibration & constants ──────────────────────────────────────────────
play_w, play_h = BBOX["width"], BBOX["height"]
GROUND_OFF     = 40          # px above paddle
Y_MIN, Y_MAX   = play_h*0.65, play_h*0.75
SAMPLES, DT    = 10, 0.0025

sct = mss.mss()

def grab():
    return np.array(sct.grab(BBOX))

#── 2) Arm on first 'P' press ────────────────────────────────────────────────
cv2.namedWindow("Arm", cv2.WINDOW_NORMAL)
cv2.imshow("Arm", np.zeros((1,1), np.uint8))
print("🔔 Press 'P' to arm detection, then switch to your game window.")
while True:
    if (cv2.waitKey(50) & 0xFF) == ord('p'):
        break
cv2.destroyWindow("Arm")

#── 3) Wait for downward pass through the 65%–75% band ──────────────────────
prev_y = None
print("⏳ Waiting for ball to move ↓ through the 65%–75% height band …")
while True:
    frame = grab()
    bx, by = find_ball(frame)
    if bx is None:
        continue
    if prev_y is not None and by > prev_y and Y_MIN <= by <= Y_MAX:
        print(f"✅ Triggered at y={by:.1f}")
        break
    prev_y = by
    time.sleep(1/60)

#── 4) Sample a short trajectory ────────────────────────────────────────────
traj = []
last = None
for _ in range(SAMPLES):
    last = grab()
    bx, by = find_ball(last)
    traj.append((bx, by))
    time.sleep(DT)

#── 5) Simulate bounce to compute final X on the slider-top (ground_y) ──────
_, paddle_y = find_paddle(last)
ground_y    = paddle_y - GROUND_OFF

(x0, y0), (x1, y1) = traj[-2], traj[-1]
vel = np.array([x1 - x0, y1 - y0], float)
pos = np.array([x1, y1], float)
segs = []
w = play_w

while vel[1] > 0:
    t_s = (ground_y - pos[1]) / vel[1]
    t_w = ((w if vel[0] > 0 else 0) - pos[0]) / vel[0]

    if 0 <= t_s <= t_w:
        end = pos + vel * t_s
        segs.append(end)
        break
    if t_w < 0:
        break

    hit = pos + vel * t_w
    segs.append(hit)
    vel[0] *= -1
    pos = hit

if segs:
    final_x = segs[-1][0]
else:
    # fallback analytic prediction
    final_x = mirror_bounce((x1, y1), ground_y)

#── 6) Print only the final magenta-dot X and exit ─────────────────────────
print(f"🎯 Magenta dot X = {final_x:.1f}")
