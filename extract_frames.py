from pathlib import Path
import cv2

VIDEO_PATH = Path("data/video.mp4")
FRAMES_DIR = Path("frames")
STEP = 10  # keep every 10th frame

def main():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % STEP == 0:
            out_path = FRAMES_DIR / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {FRAMES_DIR}")

if __name__ == "__main__":
    main()
