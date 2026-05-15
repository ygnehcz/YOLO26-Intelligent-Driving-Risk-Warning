"""
基于 YOLO26 的道路目标检测与风险预警 — 逐帧视频推理脚本

    阶段 3.5：在阶段 3 基础上增加近距视觉约束（bbox 高度比阈值）校准预警逻辑。
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── 风险预警模块 ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.warning_utils import (
    MIN_BBOX_HEIGHT_RATIO,
    RISK_CLASS_NAMES,
    detect_risk_vehicles,
    draw_risk_zone,
    draw_risk_vehicle_boxes,
    draw_warning_banner,
    get_risk_zone_polygon,
)


# ── 项目根目录（scripts/ 的父目录）──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 默认路径 ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_videos" / "road_drive_01.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "videos" / "road_drive_01_risk_warning.mp4"

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


def process_frame(frame, model: YOLO, risk_polygon: list):
    """
    对单帧执行检测并叠加风险预警标注。

    流程：
        1. YOLO 检测 → 2. 筛选风险车辆 → 3. 绘制风险区域和预警信息

    返回：(标注后的帧, 检测结果, warning_active)

    说明：
        在阶段 2 预留的扩展点上进行了完整实现。
        后续可继续在此函数中增加 TTC 计算、行人/骑行者风险提示等。
    """
    results = model(frame, verbose=False)
    result = results[0]

    # 1. YOLO 基础标注
    annotated = result.plot()

    # 2. 风险车辆检测（含近距视觉约束）
    risk_vehicles = detect_risk_vehicles(result, RISK_CLASS_NAMES, risk_polygon,
                                         frame_height=frame.shape[0])
    warning_active = len(risk_vehicles) > 0

    # 3. 绘制风险区域（有预警/无预警不同样式）
    draw_risk_zone(annotated, risk_polygon, warning_active)

    # 4. 对风险车辆额外醒目标记
    if warning_active:
        draw_risk_vehicle_boxes(annotated, risk_vehicles)
        draw_warning_banner(annotated)

    return annotated, result, warning_active


def run_detection(
    model_path: Path = DEFAULT_MODEL,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    """主检测流程：逐帧读取 → 检测 → 风险预警 → 写入输出视频。"""
    print(f"[INFO] === 阶段 3：启用车辆风险区域预警 ===")

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
    print(f"[INFO] Risk rule: point in polygon + bbox height ratio >= {MIN_BBOX_HEIGHT_RATIO}")

    ensure_output_dir(output_path)
    writer = create_video_writer(output_path, info["width"], info["height"], info["fps"])

    frame_idx = 0
    warning_frame_count = 0
    print(f"[INFO] 开始逐帧检测（含风险预警）...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _, warning_active = process_frame(frame, model, risk_polygon)
        writer.write(annotated)

        if warning_active:
            warning_frame_count += 1

        frame_idx += 1
        if frame_idx % LOG_INTERVAL == 0:
            pct = (frame_idx / info["total_frames"] * 100) if info["total_frames"] > 0 else 0
            print(f"[INFO] 进度：{frame_idx}/{info['total_frames']} 帧（{pct:.1f}%）")

    cap.release()
    writer.release()

    # ── 汇总统计 ──────────────────────────────────────────────────────────────
    warning_pct = (warning_frame_count / frame_idx * 100) if frame_idx > 0 else 0
    print(f"[INFO] === 检测完成 ===")
    print(f"[INFO] 总处理帧数：{frame_idx}")
    print(f"[INFO] 风险预警帧数：{warning_frame_count}")
    print(f"[INFO] 风险预警帧占比：{warning_pct:.1f}%")
    print(f"[INFO] 输出视频已保存至：{output_path.resolve()}")


def main():
    """命令行入口。"""
    # 支持命令行覆盖默认路径（可选扩展）
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
