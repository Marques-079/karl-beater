# trigger_predict.py

import cv2
import mss
import numpy as np
import time

from calc_final import find_ball, find_paddle, mirror_bounce, BBOX

#── 1) Calibration & constants ──────────────────────────────────────────────
TL_x, TL_y = BBOX["left"], BBOX["top"]
play_w  = BBOX["width"]
play_h  = BBOX["height"]

# 12% in from left/right
LEFT_TRIG  = play_w * 0.12
RIGHT_TRIG = play_w * 0.88

# sampling
SAMPLES     = 10
SAMPLE_DT   = 0.0025  # sec

# ball radius → adjusted bounce
BALL_RADIUS = 14.8
LEFT_BOUND  = BALL_RADIUS
RIGHT_BOUND = play_w - BALL_RADIUS

# ground line 40px above paddle
GROUND_OFF  = 40

# colors (BGR)
RED    = (0,   0, 255)
NEON   = (57,255, 20)
BLACK  = (0,   0,   0)
PURPLE = (255,  0, 255)

sct = mss.mss()

def grab_raw():
    """Grab calibrated window as BGRA."""
    return np.array(sct.grab(BBOX))

def mirror_bounce_adj(x):
    """Reflect around [LEFT_BOUND, RIGHT_BOUND]."""
    span = RIGHT_BOUND - LEFT_BOUND
    pos  = x - LEFT_BOUND
    m = pos % (2*span)
    xr = m if m <= span else (2*span - m)
    return xr + LEFT_BOUND

#── 2) Instruction window ───────────────────────────────────────────────────
cv2.namedWindow("Press P to Trigger", cv2.WINDOW_NORMAL)
instr = np.zeros((100,400,3),dtype=np.uint8)
cv2.putText(instr,
            "Press 'P' when ball crosses 12% moving ↓",
            (10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

print("Focus this window and press 'P' at the right moment.")
print("Esc to quit.")

prev_x = None

try:
    while True:
        # show instructions
        cv2.imshow("Press P to Trigger", instr)
        key = cv2.waitKey(30) & 0xFF

        #
        if key == ord('p'):
            # ─── WAIT FOR FIRST VALID CROSSING ──────────────────────────
            print("⏳ Waiting for 12% crossing ↓ …")
            prev_x, prev_y = None, None
            while True:
                raw0 = grab_raw()

                bx0, by0 = find_ball(raw0)
                if prev_x is not None:
                    downward = (by0 > prev_y)
                    crossed  = ((prev_x < LEFT_TRIG <= bx0) or
                                (prev_x > RIGHT_TRIG >= bx0))
                    if crossed and downward:
                        print("✅ Triggered at x=", bx0, "y=", by0)
                        break
                prev_x, prev_y = bx0, by0
                time.sleep(1/60)

            # ─── NOW SAMPLE YOUR TRAJECTORY ─────────────────────────────
            traj = []
            last = None
            for _ in range(SAMPLES):
                raw = grab_raw()
                last = raw
                bx,by = find_ball(raw)
                traj.append((bx,by))
                time.sleep(SAMPLE_DT)

            # ─── REST OF YOUR CODE FOLLOWS UNCHANGED ────────────────────
            _, paddle_y = find_paddle(last)
            ground_y    = paddle_y - GROUND_OFF

            disp = last[:,:, :3].copy()
            h, w = disp.shape[:2]

            # Red polyline + dots
            pts = np.array(traj, dtype=np.int32)
            cv2.polylines(disp, [pts], False, RED, 2)
            for x,y in traj:
                cv2.circle(disp,(int(x),int(y)),3,RED,-1)

            # Neon rebound …
            x0,y0 = traj[-2]; x1,y1 = traj[-1]
            vel = np.array([x1-x0, y1-y0],float)
            pos = np.array([x1,y1],float)
            segs = []
            while True:
                if vel[1] <= 0: break
                t_p = (paddle_y - pos[1]) / vel[1]
                t_w = vel[0]>0 and (w-pos[0])/vel[0] or (0-pos[0])/vel[0]
                wall_x = w if vel[0]>0 else 0

                if 0 <= t_p <= t_w:
                    end = pos + vel*t_p
                    segs.append((pos.copy(),end.copy()))
                    break
                if t_w < 0: break
                hit = pos + vel*t_w
                segs.append((pos.copy(),hit.copy()))
                vel[0] *= -1
                pos = hit

            for a,b in segs:
                cv2.line(disp, tuple(map(int,a)), tuple(map(int,b)), NEON, 2, cv2.LINE_AA)

            # Black ground line + purple dot
            final_x = int(segs[-1][1][0])
            cv2.line(disp,(0,int(ground_y)),(w,int(ground_y)),BLACK,2)
            cv2.circle(disp,(final_x,int(ground_y)),6,PURPLE,-1)

            cv2.putText(disp,f"Land X={final_x}",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,NEON,2)

            cv2.imshow("Prediction", disp)
            cv2.waitKey(0)
            cv2.destroyWindow("Prediction")


finally:
    cv2.destroyAllWindows()
