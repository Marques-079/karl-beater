import mss, numpy as np, time

TL_x, TL_y = 19 + 14.8,  185
BR_x, BR_y = 335 - 14.8 , 750

BBOX = {
    "left":   TL_x,
    "top":    TL_y,
    "width":  BR_x - TL_x,
    "height": BR_y - TL_y
}

TARGET_BALL_BGR = np.array([37, 48, 193])  # BGR format
BALL_TOL = (TARGET_BALL_BGR * 0.20).astype(int)
PADDLE_MAX_BGR = np.array([10,10,10])

sct = mss.mss()
play_w = BBOX["width"]

# ── 1) Simple Kalman Filter for [x, y, vx, vy] ─────────────────────────────
class Kalman2D:
    def __init__(self, dt=1.0):
        # State: [x, y, vx, vy]
        self.dt = dt
        # State transition
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1, 0],
                           [0,0,0, 1]], dtype=float)
        # Measurement: we only measure [x, y]
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float)
        # Covariances
        self.P = np.eye(4)*500.     # large initial uncertainty
        self.Q = np.eye(4)*1e-1     # process noise
        self.R = np.eye(2)*5.       # measurement noise
        # initial state
        self.x = np.zeros((4,1))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, meas):
        """
        meas: (2,) array = [x_measured, y_measured]
        """
        z = meas.reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x

kf = Kalman2D(dt=1.0)

# ── 2) Helpers ─────────────────────────────────────────────────────────────
def grab_raw():
    return np.array(sct.grab(BBOX))

def find_ball(raw):
    bgr = raw[:,:,:3].astype(int)
    diff = np.abs(bgr - TARGET_BALL_BGR[None,None,:])
    mask = np.all(diff <= BALL_TOL[None,None,:], axis=2)
    ys, xs = np.where(mask)
    if xs.size==0:
        raise RuntimeError("Ball not found")
    return xs.mean(), ys.mean()

def find_paddle(raw):
    bgr = raw[:,:,:3]
    strip = bgr[-30:,:,:]
    mask  = np.all(strip <= PADDLE_MAX_BGR[None,None,:], axis=2)
    ys, xs = np.where(mask)
    if xs.size==0:
        raise RuntimeError("Paddle not found")
    y_full = raw.shape[0] - 30 + ys.mean()
    return xs.mean(), y_full

def mirror_bounce(x):
    period = 2*play_w
    m = x % period
    return m if m<=play_w else period-m
    

# ── 3) Prediction + console demo ───────────────────────────────────────────
if __name__=="__main__":
    try:
        print("Running Kalman‐smoothed landing predictor. Ctrl+C to quit.")
        while True:
            raw = grab_raw()
            bx, by = find_ball(raw)
            px, py = find_paddle(raw)

            # 1) predict state forward
            kf.predict()

            # 2) correct with current measurement
            state = kf.update(np.array([bx, by]))

            # unpack filtered
            x_f, y_f, vx, vy = state.flatten()

            # 3) if ball is rising, just use paddle center
            if vy <= 0:
                land_x = px
            else:
                # time in frames to hit paddle_y
                dt = (py - y_f)/vy
                raw_x = x_f + vx*dt
                land_x = int(mirror_bounce(raw_x))

            # map to screen if needed
            screen_x = int(TL_x + land_x)

            print(f"Measured ball @ ({bx:.1f},{by:.1f}) → "
                  f"Filtered vel=({vx:.1f},{vy:.1f}) → "
                  f"Land @ x={land_x} (screen {screen_x})")

            time.sleep(1/30)
    except KeyboardInterrupt:
        print("Goodbye!")
