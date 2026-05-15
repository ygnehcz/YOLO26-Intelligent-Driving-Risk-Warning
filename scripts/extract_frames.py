"""Extract specific frames from raw and warning videos for analysis."""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_VIDEO = PROJECT_ROOT / "data" / "test_videos" / "road_drive_01.mp4"
WARNING_VIDEO = PROJECT_ROOT / "outputs" / "videos" / "road_drive_01_risk_warning.mp4"
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis_samples"

FRAMES_TO_EXTRACT = [1, 100, 200, 300, 400, 500, 600, 700]


def extract_frames(video_path: Path, prefix: str) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {prefix} — total frames: {total}")

    frame_idx = 0
    extracted = 0
    next_target_idx = 0

    while next_target_idx < len(FRAMES_TO_EXTRACT):
        ret, frame = cap.read()
        if not ret:
            break

        target_frame_no = FRAMES_TO_EXTRACT[next_target_idx]
        # target_frame_no is 1-indexed; frame_idx is 0-indexed
        if frame_idx == target_frame_no - 1:
            out_path = OUT_DIR / f"{prefix}_frame_{target_frame_no:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"  Saved: {out_path.name}")
            extracted += 1
            next_target_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] {prefix} — extracted {extracted} frames")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUT_DIR}")
    extract_frames(RAW_VIDEO, "raw")
    extract_frames(WARNING_VIDEO, "warning")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
