import mss, numpy as np, time
from calc import find_ball, find_paddle, BBOX

TL_x, TL_y = 19,  185
BR_x, BR_y = 335, 750

BBOX = {
    "left":   TL_x,
    "top":    TL_y,
    "width":  BR_x - TL_x,
    "height": BR_y - TL_y
}

TARGET_BALL_BGR = np.array([37, 48, 193])
BALL_TOL = (TARGET_BALL_BGR * 0.10).astype(int)
PADDLE_MAX_BGR = np.array([10,10,10])

sct    = mss.mss()
play_w = BBOX["width"]
play_h = BBOX["height"]

# ── NEW PARAM: Ball radius in pixels ──────────────────────────────────────
BALL_RADIUS = 14.8
LEFT_BOUND  = BALL_RADIUS
RIGHT_BOUND = play_w - BALL_RADIUS


def grab_raw():
    return np.array(sct.grab(BBOX))


def find_ball(raw):
    bgr  = raw[:,:,:3].astype(int)
    diff = np.abs(bgr - TARGET_BALL_BGR[None,None,:])
    mask = np.all(diff <= BALL_TOL[None,None,:], axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise RuntimeError("Ball not found")
    return xs.mean(), ys.mean()


def find_paddle(raw):
    bgr   = raw[:,:,:3]
    strip = bgr[-30:,:,:]
    mask  = np.all(strip <= PADDLE_MAX_BGR[None,None,:], axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise RuntimeError("Paddle not found")
    y_full = raw.shape[0] - 30 + ys.mean()
    return xs.mean(), y_full


# ── UPDATED bounce function using ball radius ─────────────────────────────
def mirror_bounce(x):
    """
    Reflect x around the invisible walls at LEFT_BOUND and RIGHT_BOUND.
    """
    span = RIGHT_BOUND - LEFT_BOUND
    # shift so boundaries map to [0, span]
    pos = x - LEFT_BOUND
    # fold via modulo 2*span
    m = pos % (2 * span)
    if m <= span:
        xr = m
    else:
        xr = 2 * span - m
    return xr + LEFT_BOUND


# ── Example: simple velocity‐based landing test ────────────────────────────
if __name__ == "__main__":
    print("Testing bounce with BALL_RADIUS =", BALL_RADIUS)
    try:
        while True:
            raw = grab_raw()
            bx, by = find_ball(raw)
            px, py = find_paddle(raw)

            # compute instantaneous vel from two quick frames
            # grab a second for the delta
            time.sleep(0.03)
            raw2 = grab_raw()
            bx2, by2 = find_ball(raw2)
            vx = bx2 - bx
            vy = by2 - by

            # if moving downward, project to paddle with bounces
            if vy > 0:
                # time to hit paddle
                dt = (py - by2) / vy
                raw_lx = bx2 + vx * dt
                land_x = int(mirror_bounce(raw_lx))
            else:
                land_x = int(px)

            screen_x = int(TL_x + land_x)
            print(f"Raw impact x={bx2:.1f}, vy={vy:.1f} → Land @ x={land_x} (screen {screen_x})")

            time.sleep(1/30)
    except KeyboardInterrupt:
        print("Done.")
