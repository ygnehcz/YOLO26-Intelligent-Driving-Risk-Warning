"""
基于 YOLO26 + ByteTrack 的多目标跟踪与稳定风险预警脚本

    阶段 6：在 ByteTrack 跟踪基础上增加基于 Track ID 累积帧数的稳定风险过滤，
    减少单帧误检和短暂闪烁。同一目标在风险区累计 >= STABLE_RISK_MIN_FRAMES 帧
    后才触发"稳定风险预警"。
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "videos" / "road_drive_01_tracked_stable_warning.mp4"
DEFAULT_TRACKER = "bytetrack.yaml"

# ── 稳定风险阈值 ────────────────────────────────────────────────────────────────
STABLE_RISK_MIN_FRAMES = 5   # Track ID 累计风险帧数 >= 此值才触发稳定风险

LOG_INTERVAL = 50
BOX_COLOR = (0, 255, 0)
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

    w = frame.shape[1]
    font_scale = max(0.4, w / 1600.0)
    thickness = max(1, int(font_scale * 1.5))

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        track_id = int(box.id[0]) if box.id is not None else None
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if track_id is not None and track_id in risk_vehicle_ids:
            continue
        if track_id is not None and track_id in risk_vru_ids:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{cls_name} ID:{track_id}" if track_id is not None else cls_name
        (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1 - 2), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, font_scale, (0, 0, 0), thickness)


def build_stable_banner(has_stable_vehicle: bool, has_stable_vru: bool,
                        has_raw_vehicle: bool, has_raw_vru: bool) -> str | None:
    """生成稳定风险优先的横幅文案。"""
    if has_stable_vehicle and has_stable_vru:
        return "STABLE WARNING: VEHICLE AND VRU RISK"
    elif has_stable_vehicle:
        return "STABLE WARNING: VEHICLE RISK"
    elif has_stable_vru:
        return "STABLE WARNING: VRU RISK"
    # 有原始风险但未达稳定阈值 — 降级横幅
    elif has_raw_vehicle and has_raw_vru:
        return "WARNING: VEHICLE AND VRU IN RISK ZONE"
    elif has_raw_vehicle:
        return "WARNING: VEHICLE IN RISK ZONE"
    elif has_raw_vru:
        return "WARNING: VRU IN RISK ZONE"
    return None


def process_frame_with_track(frame, model: YOLO, risk_polygon: list, tracker: str,
                              risk_frame_counter: dict):
    """
    对单帧执行跟踪检测，叠加双类风险预警标注，支持稳定风险判定。

    稳定风险：同一 Track ID 在风险区累计帧数 >= STABLE_RISK_MIN_FRAMES。

    返回：(标注帧, raw_vehicle, raw_vru, stable_vehicle, stable_vru,
           risk_vehicle_ids, risk_vru_ids, stable_v_ids, stable_vru_ids)
    """
    results = model.track(frame, persist=True, tracker=tracker, verbose=False)
    result = results[0]
    frame_h = frame.shape[0]

    annotated = frame.copy()

    # 1. 检测风险目标
    risk_vehicles = detect_risk_targets(
        result, VEHICLE_CLASSES, risk_polygon, frame_h, MIN_VEHICLE_BBOX_HEIGHT_RATIO)
    risk_vrus = detect_risk_targets(
        result, VRU_CLASSES, risk_polygon, frame_h, MIN_VRU_BBOX_HEIGHT_RATIO)

    # 2. 更新各 Track ID 的累计风险帧数
    for v in risk_vehicles:
        tid = v.get("track_id")
        if tid is not None:
            risk_frame_counter[tid] += 1
    for v in risk_vrus:
        tid = v.get("track_id")
        if tid is not None:
            risk_frame_counter[tid] += 1

    # 3. 区分稳定风险与原始风险
    stable_vehicles = [v for v in risk_vehicles
                       if v.get("track_id") is not None
                       and risk_frame_counter[v["track_id"]] >= STABLE_RISK_MIN_FRAMES]
    stable_vrus = [v for v in risk_vrus
                   if v.get("track_id") is not None
                   and risk_frame_counter[v["track_id"]] >= STABLE_RISK_MIN_FRAMES]

    has_raw_vehicle = len(risk_vehicles) > 0
    has_raw_vru = len(risk_vrus) > 0
    has_stable_vehicle = len(stable_vehicles) > 0
    has_stable_vru = len(stable_vrus) > 0
    warning_active = has_raw_vehicle or has_raw_vru

    risk_vehicle_ids = {v["track_id"] for v in risk_vehicles if "track_id" in v}
    risk_vru_ids = {v["track_id"] for v in risk_vrus if "track_id" in v}
    stable_v_ids = {v["track_id"] for v in stable_vehicles if "track_id" in v}
    stable_vru_ids = {v["track_id"] for v in stable_vrus if "track_id" in v}

    # 4. 绘制普通检测框
    draw_detection_boxes(annotated, result, risk_vehicle_ids, risk_vru_ids)

    # 5. 绘制风险区域
    draw_risk_zone(annotated, risk_polygon, warning_active)

    # 6. 绘制风险目标标记（稳定风险更醒目）
    if warning_active:
        # 不稳定风险车辆 — 原始红色框
        unstable_v = [v for v in risk_vehicles if v not in stable_vehicles]
        if unstable_v:
            draw_risk_vehicle_boxes(annotated, unstable_v)
        # 稳定风险车辆 — STABLE 标签
        if stable_vehicles:
            draw_risk_vehicle_boxes(annotated, stable_vehicles, label_prefix="STABLE RISK VEHICLE")

        # 不稳定 VRU — 原品红框
        unstable_vru = [v for v in risk_vrus if v not in stable_vrus]
        if unstable_vru:
            draw_risk_vru_boxes(annotated, unstable_vru)
        # 稳定 VRU — STABLE 标签
        if stable_vrus:
            draw_risk_vru_boxes(annotated, stable_vrus, label_prefix="STABLE RISK VRU")

        # 横幅
        banner = build_stable_banner(has_stable_vehicle, has_stable_vru,
                                     has_raw_vehicle, has_raw_vru)
        if banner:
            draw_warning_banner(annotated, banner)

    return (annotated, has_raw_vehicle, has_raw_vru,
            has_stable_vehicle, has_stable_vru,
            risk_vehicle_ids, risk_vru_ids, stable_v_ids, stable_vru_ids)


# ── 主流程 ──────────────────────────────────────────────────────────────────────

def run_tracking(
    model_path: Path = DEFAULT_MODEL,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    tracker: str = DEFAULT_TRACKER,
) -> None:
    print(f"[INFO] === 阶段 6：ByteTrack 跟踪 + 稳定风险过滤 ===")

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
    print(f"[INFO] 稳定风险最小帧数：{STABLE_RISK_MIN_FRAMES}")
    print(f"[INFO] Vehicle risk rule: point in polygon + bbox height ratio >= {MIN_VEHICLE_BBOX_HEIGHT_RATIO}")
    print(f"[INFO] VRU risk rule:     point in polygon + bbox height ratio >= {MIN_VRU_BBOX_HEIGHT_RATIO}")

    ensure_output_dir(output_path)
    writer = create_video_writer(output_path, info["width"], info["height"], info["fps"])

    frame_idx = 0
    raw_vehicle_count = 0
    raw_vru_count = 0
    raw_any_count = 0
    stable_vehicle_count = 0
    stable_vru_count = 0
    stable_any_count = 0
    all_track_ids = set()
    vehicle_risk_id_frames = defaultdict(int)
    vru_risk_id_frames = defaultdict(int)
    risk_frame_counter = defaultdict(int)   # track_id → 累计风险帧数

    print(f"[INFO] 开始逐帧跟踪检测...")

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break

        (annotated, has_raw_v, has_raw_vru,
         has_stable_v, has_stable_vru,
         risk_v_ids, risk_vru_ids_set,
         stable_v_ids, stable_vru_ids) = \
            process_frame_with_track(frame, model, risk_polygon, tracker,
                                     risk_frame_counter)

        writer.write(annotated)

        # 原始风险统计
        if has_raw_v:
            raw_vehicle_count += 1
        if has_raw_vru:
            raw_vru_count += 1
        if has_raw_v or has_raw_vru:
            raw_any_count += 1

        # 稳定风险统计
        if has_stable_v:
            stable_vehicle_count += 1
        if has_stable_vru:
            stable_vru_count += 1
        if has_stable_v or has_stable_vru:
            stable_any_count += 1

        # 风险 ID 帧数追踪
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
    raw_any_pct = (raw_any_count / frame_idx * 100) if frame_idx > 0 else 0
    stable_any_pct = (stable_any_count / frame_idx * 100) if frame_idx > 0 else 0

    print(f"\n[INFO] === 跟踪检测完成 ===")
    print(f"[INFO] 总处理帧数：              {frame_idx}")
    print(f"[INFO] 不同 Track ID 总数：      {len(all_track_ids)}")
    print(f"[INFO] --- 原始风险统计 ---")
    print(f"[INFO] 原始车辆风险帧数：        {raw_vehicle_count}")
    print(f"[INFO] 原始 VRU 风险帧数：       {raw_vru_count}")
    print(f"[INFO] 原始任一风险帧数：        {raw_any_count}")
    print(f"[INFO] 原始任一风险占比：        {raw_any_pct:.1f}%")
    print(f"[INFO] --- 稳定风险统计（>= {STABLE_RISK_MIN_FRAMES} 帧）---")
    print(f"[INFO] 稳定车辆风险帧数：        {stable_vehicle_count}")
    print(f"[INFO] 稳定 VRU 风险帧数：       {stable_vru_count}")
    print(f"[INFO] 稳定任一风险帧数：        {stable_any_count}")
    print(f"[INFO] 稳定任一风险占比：        {stable_any_pct:.1f}%")

    # 稳定风险 ID 统计（只列累计 >= STABLE_RISK_MIN_FRAMES 的 ID）
    stable_vehicle_ids = {tid: c for tid, c in vehicle_risk_id_frames.items()
                          if c >= STABLE_RISK_MIN_FRAMES}
    stable_vru_ids = {tid: c for tid, c in vru_risk_id_frames.items()
                      if c >= STABLE_RISK_MIN_FRAMES}

    if stable_vehicle_ids:
        print(f"\n[INFO] 稳定车辆风险 ID 统计：")
        for tid in sorted(stable_vehicle_ids.keys()):
            print(f"      车辆 ID {tid:3d}: {stable_vehicle_ids[tid]:4d} 帧")

    if stable_vru_ids:
        print(f"\n[INFO] 稳定 VRU 风险 ID 统计：")
        for tid in sorted(stable_vru_ids.keys()):
            print(f"      VRU  ID {tid:3d}: {stable_vru_ids[tid]:4d} 帧")

    print(f"\n[INFO] 输出视频已保存至：{output_path.resolve()}")


# ── 命令行入口 ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO26 + ByteTrack + 稳定风险预警")
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
