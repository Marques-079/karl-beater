import cv2
import mss
import numpy as np
import time
import pyautogui

from calc_final import find_ball, find_paddle, mirror_bounce, BBOX

# ── Calibration & constants ────────────────────────────────────────────────
last_move, rebound,  contacts = None, 1, 0
play_w, play_h    = BBOX["width"], BBOX["height"]
 # samples for trajectory fit
SAMPLE_DT = 0.000000001  # sec between samples
SAMPLES   = 15

BALL_RADIUS = 14.5
LEFT_BOUND  = BALL_RADIUS
RIGHT_BOUND = play_w - BALL_RADIUS

ground_offset = 40  # slider‐top line above paddle

# 65%–75% vertical band where we want to catch the ball moving down
#DIMENSIONS ARE TKAN FROM TOP LEFT (POS, POS)
Y_MIN = play_h * 0.20
Y_MAX = play_h * 0.32

sct = mss.mss()

def grab_raw():
    """Grab the calibrated play area as a BGRA image."""
    return np.array(sct.grab(BBOX))

def mirror_bounce_adj(x):
    """Analytic reflection of x between LEFT_BOUND and RIGHT_BOUND."""
    span = RIGHT_BOUND - LEFT_BOUND
    pos  = x - LEFT_BOUND
    m    = pos % (2 * span)
    xr   = m if m <= span else (2 * span - m)
    return xr + LEFT_BOUND

cv2.namedWindow("Trigger", cv2.WINDOW_NORMAL)

run_detection = False   # becomes True after first 'P' press
screen_y      = None    # will hold the paddle-center Y on screen

try:
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

        # ── On first 'p', grab paddle position and prepare for dragging ───
        if not run_detection and key == ord("p"):
            pyautogui.click(BBOX["left"] + 10, BBOX["top"] + 10)
            time.sleep(0.2)

            frame = grab_raw()
            px, py = find_paddle(frame)
            if px is None:
                continue

            screen_x = BBOX["left"] + px - 150
            screen_y = BBOX["top"]  + py - 720  # calibrated offset

            pyautogui.moveTo(screen_x, screen_y, duration=0.002)
            print(f"🟢 Paddle at screen X={screen_x:.1f}, screen Y={screen_y:.1f}")
            run_detection = True

        if not run_detection:
            continue

        # ── 1) Wait for the ball moving down into our catch band ─────────
        prev_y = None
        while True:
            raw0 = grab_raw()
            bx0, by0 = find_ball(raw0)

            
            if bx0 is None:
                continue
            if prev_y is not None and by0 > prev_y and (Y_MIN <= by0 <= Y_MAX):
                print(f"⏳ Trigger at ball ({bx0:.1f},{by0:.1f})")
                break

            prev_y = by0
        
            #time.sleep(0.002)

        # ── 2) Sample a short trajectory ─────────────────────────────────
        traj = []
        last = None

        def contact():
            global contacts
            if contacts >= 15:
                print("Resetting contacts")
                
                contacts += 1
                return 8
            else:
                contacts += 1
                return 15

        print(f"This is contact number {contacts} with the ball")

        for i in range(contact()):
            #start = time.perf_counter()
            raw = grab_raw()
            last = raw
            bx, by = find_ball(raw)

            #Prevent wall bad samples
            if len(traj) > 5 and (bx < 40 or bx > 510):
               break
            #if bx > play_h * 0.80:
            #    break
            traj.append((bx, by))
            #end = time.perf_counter()
            #print(f"Elapsed (high-res): {end - start:.6f} seconds")
            #print(f"  Sample {i}: ball at ({bx:.1f},{by:.1f})")
            #time.sleep(SAMPLE_DT)

        # ── 3a) Build bounce segments ───────────────────────────────────────
        _, paddle_y = find_paddle(last)
        ground_y    = BBOX["top"] + paddle_y - 385
        x0, y0      = traj[-2]
        x1, y1      = traj[-1]
        vel = np.array([x1-x0, y1-y0], float)
        pos = np.array([x1,    y1],    float)
        w_img = last.shape[1]

        segs = []
        debug_lines = []
        step = 0
        # record initial state
        debug_lines.append(f"Step {step}: pos=({pos[0]:.1f},{pos[1]:.1f}), "
                           f"vel=({vel[0]:.2f},{vel[1]:.2f})")
        while vel[1] > 0:
            t_s = (ground_y - pos[1]) / vel[1]
            t_w = ((w_img if vel[0]>0 else 0) - pos[0]) / vel[0]
            debug_lines.append(f"  t_s={t_s:.2f}, t_w={t_w:.2f}")

            if 0 <= t_s <= t_w:
                end = pos + vel * t_s
                segs.append((pos.copy(), end.copy()))
                debug_lines.append(f"  → hits slider-plane at ({end[0]:.1f},{end[1]:.1f})")
                break

            if t_w < 0:
                debug_lines.append("  → wall-hit time negative → abort")
                break

            hit = pos + vel * t_w
            segs.append((pos.copy(), hit.copy()))
            debug_lines.append(f"  → bounces at ({hit[0]:.1f},{hit[1]:.1f})")
            vel[0] *= -1
            pos = hit.copy()
            step += 1
            debug_lines.append(f"Step {step}: pos=({pos[0]:.1f},{pos[1]:.1f}), "
                               f"vel=({vel[0]:.2f},{vel[1]:.2f})")

        # compute the intersection-X
        if segs:
            ix, iy = segs[-1][1]
        else:
            ix = mirror_bounce_adj(x1)
            iy = ground_y
            debug_lines.append(f"fallback analytic X={ix:.1f}")
        final_x = int(ix)
        debug_lines.append(f"Final X = {final_x}")

        # ── 3b) Build the overlay image ───────────────────────────────────
        disp = last[:, :, :3].copy()          # drop alpha
        h, w = disp.shape[:2]

        # draw past trajectory (red)
        pts = np.array(traj, dtype=np.int32)
        cv2.polylines(disp, [pts], False, (0,0,255), 2)
        for bx, by in traj:
            cv2.circle(disp, (int(bx), int(by)), 3, (0,0,255), -1)

        # neon‐green bounce segments & slider line
        NEON    = (57,255,20)
        MAGENTA = (255,0,255)
        for a, b in segs:
            cv2.line(disp,
                     (int(a[0]), int(a[1])),
                     (int(b[0]), int(b[1])),
                     NEON, 2, cv2.LINE_AA)

        # horizontal slider-plane
        cv2.line(disp, (0,int(ground_y)), (w,int(ground_y)), NEON, 2, cv2.LINE_AA)
        # magenta intersection dot
        print(f"Relative X is : {final_x}")
        cv2.circle(disp, (final_x, int(ground_y)), 6, MAGENTA, -1) #IMPORTANT HEREEE

        
        # ── 3c) Render debug text down the right margin ─────────────────────────────────
        font     = cv2.FONT_HERSHEY_SIMPLEX
        scale    = 0.5
        thickness= 1
        # starting point in image coords
        tx = w - 300
        ty = 20
        for line in debug_lines:
            cv2.putText(disp, line, (tx, ty), font, scale, (255,255,255), thickness, cv2.LINE_AA)
            ty += int(20 * scale)

        # ── 3d) Save it for later inspection ───────────────────────────────
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"prediction_debug_{ts}.png"
        cv2.imwrite(fname, disp)
        print(f"Saved debug overlay → {fname}")
        
        # ── 4) Finally, drag the paddle ────────────────────────────────────
    
        special_y = screen_y = BBOX["top"]  + py - 720

        #Calibrate 
        final_x = (final_x / 565) * 318 + 14

        print(f'Recieved {final_x}')
        #pyautogui.moveTo(final_x, special_y)
        pyautogui.dragTo(final_x, special_y, duration=0.001, button='left')

#FSR the 345 which is max of our pixel range in desktop is not 346 relative to the printed frame. 
#MAKE transformation algoroth 
# Ours is 14.8 -> 335  Spans = 320.2
# Relative is 0 -> 565
except KeyboardInterrupt:
    pass

finally:
    if run_detection:
        pyautogui.mouseUp()
    cv2.destroyAllWindows()
