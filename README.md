# Arabic QCM Video OCR

Extract multiple-choice questions (QCM) from Arabic educational videos.

This project streams raw frames directly from FFmpeg, removes duplicate slides using pixel-change detection, crops the question region, and performs Arabic OCR using PaddleOCR.

## Features

- Raw frame streaming (no temporary image files)
- Intelligent duplicate slide detection
- Region masking for UI noise
- Adaptive cropping
- Arabic text extraction

## Tech Stack

- FFmpeg (rawvideo pipe)
- OpenCV
- NumPy
- PaddleOCR (Arabic)

## Usage

1. Place your video file in the project directory.
2. Adjust resolution and crop coordinates if needed.
3. Run:

```bash
python script_name.py
