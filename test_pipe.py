import subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from paddleocr import PaddleOCR

# ---------- CONFIG ----------
import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

VIDEO_PATH = Path(__file__).resolve().parent / "test.mp4"

WIDTH = 1920
HEIGHT = 1080
FRAME_SIZE = WIDTH * HEIGHT

IGNORE_Y1, IGNORE_Y2 = 800, 1080
IGNORE_X1, IGNORE_X2 = 0, 200

PIXEL_DIFF_THRESHOLD = 25
CHANGE_PERCENT_THRESHOLD = 0.05  # slightly stricter now

# ---------- OCR INIT ----------

ocr = PaddleOCR(use_textline_orientation=True, lang="ar")
# ---------- FFMPEG ----------

cmd = [
    "ffmpeg",
    "-i",
    str(VIDEO_PATH),
    "-vf",
    "fps=1",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "gray",
    "-loglevel",
    "error",
    "pipe:1",
]

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

prev_frame = None
unique_count = 0

print("Processing...")

# ---------- LOOP ----------

while True:
    raw_frame = process.stdout.read(FRAME_SIZE)

    if len(raw_frame) != FRAME_SIZE:
        break

    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH))

    frame = frame.copy()
    frame[IGNORE_Y1:IGNORE_Y2, IGNORE_X1:IGNORE_X2] = 0

    if prev_frame is not None:

        diff = cv2.absdiff(frame, prev_frame)
        changed_pixels = diff > PIXEL_DIFF_THRESHOLD
        num_changed = np.count_nonzero(changed_pixels)
        total_pixels = HEIGHT * WIDTH
        change_ratio = num_changed / total_pixels

        if change_ratio < CHANGE_PERCENT_THRESHOLD:
            continue

    unique_count += 1
    print(f"\n==============================")
    print(f"Unique frame #{unique_count}")

    # ---------- CROP QCM REGION ----------
    # ---------- CROP QCM REGION ----------
    CROP_Y1, CROP_Y2 = 200, 800
    CROP_X1, CROP_X2 = 200, 1700

    roi = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

    # Resize (important)
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to 3-channel (VERY IMPORTANT)
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # ---------- OCR ----------
    result = ocr.predict(roi)

    extracted_text = ""

    for line in result:
        try:
            extracted_text += line["rec_text"] + "\n"
        except:
            pass

    print("OCR Result:")
    print(extracted_text)

    # Optional save
    Image.fromarray(roi).save(f"frame_{unique_count}.jpg")

    prev_frame = frame.copy()

process.stdout.close()
process.wait()

print("\nTotal unique frames:", unique_count)
