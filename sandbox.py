import pyautogui
import time
import sys

def countdown(n):
    for i in range(n, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

def drag_square_moveTo(cx, cy, size=200, duration=0.5):
    """
    Click-and-hold at (cx,cy), then drag a square of side `size`
    using moveTo for each corner, then release.
    """
    print(f"Dragging square via moveTo from ({cx},{cy}), size {size}")
    # 1) Go to start
    pyautogui.moveTo(cx, cy, duration=0.2)
    # 2) Click and hold
    pyautogui.mouseDown(button='left')
    # 3) Drag to each corner
    pyautogui.moveTo(cx + size, cy, duration=duration)
    pyautogui.moveTo(cx + size, cy + size, duration=duration)
    pyautogui.moveTo(cx, cy + size, duration=duration)
    pyautogui.moveTo(cx, cy, duration=duration)
    # 4) Release
    pyautogui.mouseUp(button='left')
    print("  → square drag complete\n")
    time.sleep(1)

def main():
    # Give you time to switch to your target window
    countdown(5)

    w, h = pyautogui.size()
    print(f"Screen size detected: {w}×{h}\n")

    # Upper-left quadrant
    drag_square_moveTo(w * 0.25, h * 0.25)
    # Center
    drag_square_moveTo(w * 0.5, h * 0.5)
    # Lower-right quadrant
    drag_square_moveTo(w * 0.75, h * 0.75)

    print("All drag-square tests complete.")
    sys.exit(0)

if __name__ == "__main__":
    main()
