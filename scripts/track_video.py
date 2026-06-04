"""
基于 YOLO26 + ByteTrack 的多目标跟踪与风险预警脚本

    阶段 5：在目标检测基础上增加 ByteTrack 多目标跟踪，为每个目标分配稳定 ID，
    统计风险目标持续帧数，为后续时序预警平滑打基础。
"""

import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ── 风险预警模块 ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.warning_utils import (
    MIN_VEHICLE_BBOX_HEIGHT_RATIO,
    MIN_VRU_BBOX_HEIGHT_RATIO,
    VEHICLE_CLASSES,
    VRU_CLASSES,
    detect_risk_targets,
    draw_risk_vehicle_boxes,
    draw_risk_vru_boxes,
    draw_risk_zone,
    draw_warning_banner,
    get_risk_zone_polygon,
)

# ── 项目根目录 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 默认路径 ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_videos" / "road_drive_01.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "videos" / "road_drive_01_tracked.mp4"
DEFAULT_TRACKER = "bytetrack.yaml"

LOG_INTERVAL = 50
BOX_COLOR = (0, 255, 0)           # 绿色：普通检测框
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── 工具函数 ────────────────────────────────────────────────────────────────────

def load_model(model_path: Path) -> YOLO:
    print(f"[INFO] 正在加载模型：{model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] 模型文件不存在：{model_path}")
    model = YOLO(str(model_path))
    print(f"[INFO] 模型加载完成")
    return model


def open_video(input_path: Path) -> cv2.VideoCapture:
    print(f"[INFO] 正在打开输入视频：{input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] 输入视频文件不存在：{input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 无法打开视频文件：{input_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}


def ensure_output_dir(output_path: Path) -> None:
    output_parent = output_path.parent
    if not output_parent.exists():
        output_parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 已创建输出目录：{output_parent}")


def create_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] 无法创建输出视频文件：{output_path}")
    return writer


def draw_detection_boxes(frame: np.ndarray, result, risk_vehicle_ids: set, risk_vru_ids: set) -> None:
    """在帧上绘制所有检测框及 Track ID（风险目标跳过，由专用函数绘制）。"""
    boxes = result.boxes
    if boxes is None:
        return

    h, w = frame.shape[:2]
    font_scale = max(0.4, w / 1600.0)
    thickness = max(1, int(font_scale * 1.5))

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        track_id = int(box.id[0]) if box.id is not None else None
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        # 风险目标由专用函数绘制，此处跳过
        if track_id is not None and track_id in risk_vehicle_ids:
            continue
        if track_id is not None and track_id in risk_vru_ids:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{cls_name} ID:{track_id}" if track_id is not None else cls_name
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1 - 2), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, font_scale, (0, 0, 0), thickness)


def build_banner_text(has_vehicle: bool, has_vru: bool) -> str | None:
    if has_vehicle and has_vru:
        return "WARNING: VEHICLE AND VRU IN RISK ZONE"
    elif has_vehicle:
        return "WARNING: VEHICLE IN RISK ZONE"
    elif has_vru:
        return "WARNING: VRU IN RISK ZONE"
    return None


def process_frame_with_track(frame, model: YOLO, risk_polygon: list, tracker: str):
    """
    对单帧执行跟踪检测并叠加双类风险预警标注。

    返回：(标注后的帧, has_vehicle_risk, has_vru_risk, risk_vehicle_ids, risk_vru_ids)
    """
    results = model.track(frame, persist=True, tracker=tracker, verbose=False)
    result = results[0]
    frame_h = frame.shape[0]

    # 基础标注用空白画布 — 我们手动绘制 boxes
    annotated = frame.copy()

    # 1. 分别检测风险目标
    risk_vehicles = detect_risk_targets(
        result, VEHICLE_CLASSES, risk_polygon, frame_h, MIN_VEHICLE_BBOX_HEIGHT_RATIO)
    risk_vrus = detect_risk_targets(
        result, VRU_CLASSES, risk_polygon, frame_h, MIN_VRU_BBOX_HEIGHT_RATIO)

    has_vehicle = len(risk_vehicles) > 0
    has_vru = len(risk_vrus) > 0
    warning_active = has_vehicle or has_vru

    risk_vehicle_ids = {v["track_id"] for v in risk_vehicles if "track_id" in v}
    risk_vru_ids = {v["track_id"] for v in risk_vrus if "track_id" in v}

    # 2. 绘制普通检测框（带 ID）
    draw_detection_boxes(annotated, result, risk_vehicle_ids, risk_vru_ids)

    # 3. 绘制风险区域
    draw_risk_zone(annotated, risk_polygon, warning_active)

    # 4. 绘制风险目标醒目标记 + 横幅
    if warning_active:
        if has_vehicle:
            draw_risk_vehicle_boxes(annotated, risk_vehicles)
        if has_vru:
            draw_risk_vru_boxes(annotated, risk_vrus)
        banner_text = build_banner_text(has_vehicle, has_vru)
        if banner_text:
            draw_warning_banner(annotated, banner_text)

    return annotated, has_vehicle, has_vru, risk_vehicle_ids, risk_vru_ids


# ── 主流程 ──────────────────────────────────────────────────────────────────────

def run_tracking(
    model_path: Path = DEFAULT_MODEL,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    tracker: str = DEFAULT_TRACKER,
) -> None:
    print(f"[INFO] === 阶段 5：ByteTrack 多目标跟踪 + 风险预警 ===")

    model = load_model(model_path)
    cap = open_video(input_path)
    info = get_video_info(cap)
    print(
        f"[INFO] 视频信息 — 分辨率：{info['width']}×{info['height']}，"
        f"FPS：{info['fps']:.2f}，总帧数：{info['total_frames']}"
    )

    risk_polygon = get_risk_zone_polygon(info["width"], info["height"])
    print(f"[INFO] 风险区域已按 {info['width']}×{info['height']} 自适应设置")
    print(f"[INFO] 跟踪器：{tracker}")
    print(f"[INFO] Vehicle risk rule: point in polygon + bbox height ratio >= {MIN_VEHICLE_BBOX_HEIGHT_RATIO}")
    print(f"[INFO] VRU risk rule:     point in polygon + bbox height ratio >= {MIN_VRU_BBOX_HEIGHT_RATIO}")

    ensure_output_dir(output_path)
    writer = create_video_writer(output_path, info["width"], info["height"], info["fps"])

    frame_idx = 0
    vehicle_warning_count = 0
    vru_warning_count = 0
    any_warning_count = 0
    all_track_ids = set()
    vehicle_risk_id_frames = defaultdict(int)
    vru_risk_id_frames = defaultdict(int)

    print(f"[INFO] 开始逐帧跟踪检测...")

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break

        annotated, has_vehicle, has_vru, risk_v_ids, risk_vru_ids_set = \
            process_frame_with_track(frame, model, risk_polygon, tracker)

        writer.write(annotated)

        # 收集所有出现的 track ID
        result = model.predictor.results if hasattr(model, 'predictor') and model.predictor else None
        # 改用最后一帧的 result 来收集 ID
        if hasattr(model, 'predictor') and model.predictor is not None:
            pass  # track IDs collected via risk targets below

        if has_vehicle:
            vehicle_warning_count += 1
        if has_vru:
            vru_warning_count += 1
        if has_vehicle or has_vru:
            any_warning_count += 1

        for tid in risk_v_ids:
            vehicle_risk_id_frames[tid] += 1
            all_track_ids.add(tid)
        for tid in risk_vru_ids_set:
            vru_risk_id_frames[tid] += 1
            all_track_ids.add(tid)

        frame_idx += 1
        if frame_idx % LOG_INTERVAL == 0:
            pct = (frame_idx / info["total_frames"] * 100) if info["total_frames"] > 0 else 0
            print(f"[INFO] 进度：{frame_idx}/{info['total_frames']} 帧（{pct:.1f}%）")

    cap.release()
    writer.release()

    # ── 汇总统计 ──────────────────────────────────────────────────────────────────
    any_warning_pct = (any_warning_count / frame_idx * 100) if frame_idx > 0 else 0
    print(f"\n[INFO] === 跟踪检测完成 ===")
    print(f"[INFO] 总处理帧数：          {frame_idx}")
    print(f"[INFO] 不同 Track ID 总数：  {len(all_track_ids)}")
    print(f"[INFO] 车辆风险预警帧数：    {vehicle_warning_count}")
    print(f"[INFO] VRU 风险预警帧数：    {vru_warning_count}")
    print(f"[INFO] 任一风险预警帧数：    {any_warning_count}")
    print(f"[INFO] 任一风险预警占比：    {any_warning_pct:.1f}%")

    if vehicle_risk_id_frames:
        print(f"\n[INFO] 车辆风险目标 ID 统计：")
        for tid in sorted(vehicle_risk_id_frames.keys()):
            print(f"      车辆 ID {tid:3d}: {vehicle_risk_id_frames[tid]:4d} 帧")

    if vru_risk_id_frames:
        print(f"\n[INFO] VRU 风险目标 ID 统计：")
        for tid in sorted(vru_risk_id_frames.keys()):
            print(f"      VRU  ID {tid:3d}: {vru_risk_id_frames[tid]:4d} 帧")

    print(f"\n[INFO] 输出视频已保存至：{output_path.resolve()}")


# ── 命令行入口 ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO26 + ByteTrack 多目标跟踪与风险预警")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO 模型路径")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入视频路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出视频路径")
    parser.add_argument("--tracker", type=str, default=DEFAULT_TRACKER, help="跟踪器配置")
    args = parser.parse_args()

    try:
        run_tracking(args.model, args.input, args.output, args.tracker)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
