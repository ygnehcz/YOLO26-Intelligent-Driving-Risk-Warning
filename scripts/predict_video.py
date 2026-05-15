"""
基于 YOLO26 的道路目标检测与风险预警 — 逐帧视频推理脚本

    阶段 4：支持车辆风险 + 弱势交通参与者（VRU）风险的双类预警。
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── 风险预警模块 ────────────────────────────────────────────────────────────
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


# ── 项目根目录（scripts/ 的父目录）──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 默认路径 ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_videos" / "road_drive_01.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "videos" / "road_drive_01_multi_risk_warning.mp4"

# ── 日志间隔（帧）──────────────────────────────────────────────────────────
LOG_INTERVAL = 50


def load_model(model_path: Path) -> YOLO:
    """加载 YOLO 模型。"""
    print(f"[INFO] 正在加载模型：{model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] 模型文件不存在：{model_path}")
    model = YOLO(str(model_path))
    print(f"[INFO] 模型加载完成")
    return model


def open_video(input_path: Path) -> cv2.VideoCapture:
    """打开输入视频并校验。"""
    print(f"[INFO] 正在打开输入视频：{input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"[ERROR] 输入视频文件不存在：{input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 无法打开视频文件（可能编码不支持或文件已损坏）：{input_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """提取视频基本信息。"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}


def ensure_output_dir(output_path: Path) -> None:
    """确保输出目录存在，不存在则自动创建。"""
    output_parent = output_path.parent
    if not output_parent.exists():
        output_parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 已创建输出目录：{output_parent}")


def create_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """创建视频写入器，保持原视频分辨率和帧率。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] 无法创建输出视频文件：{output_path}")
    return writer


def build_banner_text(has_vehicle: bool, has_vru: bool) -> str | None:
    """根据风险类型组合决定预警横幅文案（优先级：both > vehicle > vru）。"""
    if has_vehicle and has_vru:
        return "WARNING: VEHICLE AND VRU IN RISK ZONE"
    elif has_vehicle:
        return "WARNING: VEHICLE IN RISK ZONE"
    elif has_vru:
        return "WARNING: VRU IN RISK ZONE"
    return None


def process_frame(frame, model: YOLO, risk_polygon: list):
    """
    对单帧执行检测并叠加车辆 + VRU 双类风险预警标注。

    流程：
        1. YOLO 检测 → 2. 分别筛选车辆/VRU 风险目标 → 3. 按优先级绘制预警信息

    返回：(标注后的帧, has_vehicle_risk, has_vru_risk)
    """
    results = model(frame, verbose=False)
    result = results[0]
    frame_h = frame.shape[0]

    # 1. YOLO 基础标注
    annotated = result.plot()

    # 2. 分别检测车辆风险与 VRU 风险
    risk_vehicles = detect_risk_targets(
        result, VEHICLE_CLASSES, risk_polygon, frame_h, MIN_VEHICLE_BBOX_HEIGHT_RATIO)
    risk_vrus = detect_risk_targets(
        result, VRU_CLASSES, risk_polygon, frame_h, MIN_VRU_BBOX_HEIGHT_RATIO)

    has_vehicle = len(risk_vehicles) > 0
    has_vru = len(risk_vrus) > 0
    warning_active = has_vehicle or has_vru

    # 3. 绘制风险区域
    draw_risk_zone(annotated, risk_polygon, warning_active)

    # 4. 绘制各类风险目标的醒目标记 + 顶部横幅
    if warning_active:
        if has_vehicle:
            draw_risk_vehicle_boxes(annotated, risk_vehicles)
        if has_vru:
            draw_risk_vru_boxes(annotated, risk_vrus)

        banner_text = build_banner_text(has_vehicle, has_vru)
        if banner_text:
            draw_warning_banner(annotated, banner_text)

    return annotated, has_vehicle, has_vru


def run_detection(
    model_path: Path = DEFAULT_MODEL,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    """主检测流程：逐帧读取 → 检测 → 车辆+VRU 风险预警 → 写入输出视频。"""
    print(f"[INFO] === 阶段 4：车辆 + VRU 双类风险预警 ===")

    model = load_model(model_path)

    cap = open_video(input_path)
    info = get_video_info(cap)
    print(
        f"[INFO] 视频信息 — 分辨率：{info['width']}×{info['height']}，"
        f"FPS：{info['fps']:.2f}，总帧数：{info['total_frames']}"
    )

    # 基于视频尺寸计算自适应风险区域
    risk_polygon = get_risk_zone_polygon(info["width"], info["height"])
    print(f"[INFO] 风险区域已按 {info['width']}×{info['height']} 自适应设置")
    print(f"[INFO] Vehicle risk rule: point in polygon + bbox height ratio >= {MIN_VEHICLE_BBOX_HEIGHT_RATIO}")
    print(f"[INFO] VRU risk rule:     point in polygon + bbox height ratio >= {MIN_VRU_BBOX_HEIGHT_RATIO}")

    ensure_output_dir(output_path)
    writer = create_video_writer(output_path, info["width"], info["height"], info["fps"])

    frame_idx = 0
    vehicle_warning_count = 0
    vru_warning_count = 0
    both_warning_count = 0
    any_warning_count = 0
    print(f"[INFO] 开始逐帧检测（含车辆 + VRU 风险预警）...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, has_vehicle, has_vru = process_frame(frame, model, risk_polygon)
        writer.write(annotated)

        if has_vehicle:
            vehicle_warning_count += 1
        if has_vru:
            vru_warning_count += 1
        if has_vehicle and has_vru:
            both_warning_count += 1
        if has_vehicle or has_vru:
            any_warning_count += 1

        frame_idx += 1
        if frame_idx % LOG_INTERVAL == 0:
            pct = (frame_idx / info["total_frames"] * 100) if info["total_frames"] > 0 else 0
            print(f"[INFO] 进度：{frame_idx}/{info['total_frames']} 帧（{pct:.1f}%）")

    cap.release()
    writer.release()

    # ── 汇总统计 ──────────────────────────────────────────────────────────────
    any_warning_pct = (any_warning_count / frame_idx * 100) if frame_idx > 0 else 0
    print(f"[INFO] === 检测完成 ===")
    print(f"[INFO] 总处理帧数：          {frame_idx}")
    print(f"[INFO] 车辆风险预警帧数：    {vehicle_warning_count}")
    print(f"[INFO] VRU 风险预警帧数：    {vru_warning_count}")
    print(f"[INFO] 车辆与 VRU 同时预警： {both_warning_count}")
    print(f"[INFO] 任一风险预警帧数：    {any_warning_count}")
    print(f"[INFO] 任一风险预警占比：    {any_warning_pct:.1f}%")
    print(f"[INFO] 输出视频已保存至：{output_path.resolve()}")


def main():
    """命令行入口。"""
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    input_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_INPUT
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUTPUT

    try:
        run_detection(model_path, input_path, output_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
