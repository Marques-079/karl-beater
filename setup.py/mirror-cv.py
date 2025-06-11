# calibrate.py
import cv2, numpy as np, pyautogui

def on_mouse(evt, x, y, flags, param):
    if evt == cv2.EVENT_LBUTTONDOWN:
        print("Mouse @", x, y)

# 1) Set up the window & mouse‐callback:
cv2.namedWindow("Click corners", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Click corners", on_mouse)

frame = None
print("→ First, bring your QuickTime mirror (with the game) to the front.")
print("→ Then focus this window and press 'x' to capture the calibration screenshot.")
print("→ After that, click the top-left & bottom-right of the GAME area.")
print("→ Press 'q' to quit.")

while True:
    # 2) If we've already captured, show it; else show a blank prompt
    if frame is not None:
        cv2.imshow("Click corners", frame)
    else:
        # create a blank image with instructions
        blank = np.zeros((200,600,3), dtype=np.uint8)
        cv2.putText(blank, "Press 'x' to capture screenshot", (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Click corners", blank)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        # 3) grab a fresh screenshot
        img = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        print("✓ Screenshot captured—now click the two corners.")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
#OVERRIDE with  - killall python in terminal if needed
