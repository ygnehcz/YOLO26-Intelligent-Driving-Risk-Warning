"""Analyze sample frames for vehicle detection metrics to assess over-warning."""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.warning_utils import (
    RISK_CLASS_NAMES,
    detect_risk_vehicles,
    get_risk_zone_polygon,
    is_point_in_polygon,
)

MODEL_PATH = PROJECT_ROOT / "yolo26n.pt"
SAMPLE_DIR = PROJECT_ROOT / "outputs" / "analysis_samples"

FRAME_NUMS = [1, 100, 200, 300, 400, 500, 600, 700]


def analyze_frame(model, frame_path: Path, frame_h: int, frame_w: int, risk_polygon: list):
    """Run YOLO on a single frame and report per-vehicle metrics."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"  [ERROR] Cannot read: {frame_path}")
        return

    results = model(frame, verbose=False)
    result = results[0]
    boxes = result.boxes

    if boxes is None:
        print(f"  No detections at all.")
        return

    risk_count = 0
    total_vehicle_count = 0

    print(f"  {'Class':>10s}  {'Conf':>6s}  {'Y2/H':>6s}  {'BboxH/H':>8s}  {'InZone':>6s}  {'Risk?':>5s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*5}")

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id].lower()
        if cls_name not in RISK_CLASS_NAMES:
            continue

        total_vehicle_count += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        bbox_h = y2 - y1
        bbox_h_ratio = bbox_h / frame_h
        y2_ratio = y2 / frame_h
        bottom_center = ((x1 + x2) / 2.0, y2)
        in_zone = is_point_in_polygon(bottom_center, risk_polygon)
        is_risk = in_zone  # current rule: only point-in-polygon

        if is_risk:
            risk_count += 1

        print(
            f"  {cls_name:>10s}  {conf:.2f}   {y2_ratio:.3f}   {bbox_h_ratio:.4f}    "
            f"{'YES' if in_zone else 'NO':>6s}  {'YES' if is_risk else 'NO':>5s}"
        )

    print(f"  ---> Vehicles: {total_vehicle_count}, Risk-triggered: {risk_count}")
    return total_vehicle_count, risk_count


def main():
    print("[INFO] Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))
    print("[INFO] Model loaded.\n")

    # Use frame 001 to get frame dimensions
    first_frame = cv2.imread(str(SAMPLE_DIR / "raw_frame_001.jpg"))
    frame_h, frame_w = first_frame.shape[:2]
    risk_polygon = get_risk_zone_polygon(frame_w, frame_h)
    print(f"[INFO] Frame size: {frame_w}x{frame_h}")
    print(f"[INFO] Risk polygon: {risk_polygon}\n")

    total_vehicles = 0
    total_risks = 0

    for fnum in FRAME_NUMS:
        raw_path = SAMPLE_DIR / f"raw_frame_{fnum:03d}.jpg"
        print(f"=== Frame {fnum:03d} ===")
        tv, tr = analyze_frame(model, raw_path, frame_h, frame_w, risk_polygon)
        if tv is not None:
            total_vehicles += tv
            total_risks += tr
        print()

    print(f"[SUMMARY] Total vehicles across samples: {total_vehicles}")
    print(f"[SUMMARY] Total risk-triggered: {total_risks}")
    print(f"[SUMMARY] Risk trigger rate: {total_risks / total_vehicles * 100:.1f}%" if total_vehicles > 0 else "")


if __name__ == "__main__":
    main()
