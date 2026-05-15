"""
基于 YOLO26 的道路目标检测 — 逐帧视频推理脚本

    阶段 2：从命令行调用升级为工程脚本，支持逐帧处理并预留风险预警扩展点。
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# ── 项目根目录（scripts/ 的父目录）──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 默认路径 ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_videos" / "road_drive_01.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "videos" / "road_drive_01_detected.mp4"

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


def process_frame(frame, model: YOLO):
    """
    对单帧执行检测。

    返回：(标注后的帧, 检测结果列表)

    说明：
        这里预留了扩展点。后续阶段可以在此函数内部对 results
        做自定义风险分析（TTC、碰撞预警等），再将信息绘制到帧上。
    """
    results = model(frame, verbose=False)
    annotated = results[0].plot()  # YOLO 内置标注
    return annotated, results[0]


def run_detection(
    model_path: Path = DEFAULT_MODEL,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    """主检测流程：逐帧读取 → 检测 → 写入输出视频。"""
    model = load_model(model_path)

    cap = open_video(input_path)
    info = get_video_info(cap)
    print(
        f"[INFO] 视频信息 — 分辨率：{info['width']}×{info['height']}，"
        f"FPS：{info['fps']:.2f}，总帧数：{info['total_frames']}"
    )

    ensure_output_dir(output_path)
    writer = create_video_writer(output_path, info["width"], info["height"], info["fps"])

    frame_idx = 0
    print(f"[INFO] 开始逐帧检测 ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = process_frame(frame, model)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % LOG_INTERVAL == 0:
            pct = (frame_idx / info["total_frames"] * 100) if info["total_frames"] > 0 else 0
            print(f"[INFO] 进度：{frame_idx}/{info['total_frames']} 帧（{pct:.1f}%）")

    cap.release()
    writer.release()

    print(f"[INFO] 检测完成，共处理 {frame_idx} 帧")
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
