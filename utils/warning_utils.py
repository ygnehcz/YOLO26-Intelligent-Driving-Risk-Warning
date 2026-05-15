"""
前方风险区域与多类交通参与者预警逻辑

    阶段 4：在阶段 3.5 基础上新增弱势交通参与者（VRU）风险提示。
    支持车辆风险与 VRU 风险的独立判定、视觉区分与优先级组合显示。
"""

import cv2
import numpy as np

# ── 目标类别分组 ──────────────────────────────────────────────────────────────
VEHICLE_CLASSES = {"car", "bus", "truck"}
VRU_CLASSES = {"person", "bicycle", "motorcycle"}

# ── 预警约束阈值（独立可调）────────────────────────────────────────────────────
MIN_VEHICLE_BBOX_HEIGHT_RATIO = 0.12   # 车辆：较严格，抑制远处小车
MIN_VRU_BBOX_HEIGHT_RATIO = 0.06       # VRU：较宽松，行人/骑行者目标更小


def get_risk_zone_polygon(frame_width: int, frame_height: int) -> list:
    """
    返回前方风险梯形区域的四个顶点（相对坐标，适配任意分辨率）。

    梯形位于画面底部中央：
        TL ───────── TR
         ╲            ╱
          BL ─────── BR
    """
    return [
        (int(0.38 * frame_width), int(0.58 * frame_height)),   # top-left
        (int(0.62 * frame_width), int(0.58 * frame_height)),   # top-right
        (int(0.92 * frame_width), int(0.98 * frame_height)),   # bottom-right
        (int(0.08 * frame_width), int(0.98 * frame_height)),   # bottom-left
    ]


def is_point_in_polygon(point: tuple, polygon: list) -> bool:
    """
    射线法判断点是否在多边形内部。

    Args:
        point: (x, y) 坐标
        polygon: [(x0,y0), (x1,y1), ...] 多边形顶点列表
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def detect_risk_targets(result, class_names: set, polygon: list,
                        frame_height: int,
                        min_bbox_height_ratio: float) -> list:
    """
    从 YOLO 检测结果中筛选落入风险区域且满足近距约束的目标。

    判定规则（两层）：
        1. 检测框底边中心点落入风险梯形区域
        2. 检测框高度 / 画面高度 >= min_bbox_height_ratio

    Args:
        result: ultralytics Results 对象
        class_names: 关注类别名称集合（小写）
        polygon: 风险区域多边形顶点
        frame_height: 画面高度（像素）
        min_bbox_height_ratio: 最小 bbox 高度占比阈值

    Returns:
        list[dict]: 每个元素包含 box (x1,y1,x2,y2), class_name, confidence
    """
    targets = []
    boxes = result.boxes
    if boxes is None:
        return targets

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id].lower()
        if cls_name not in class_names:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bottom_center = ((x1 + x2) / 2.0, y2)

        if not is_point_in_polygon(bottom_center, polygon):
            continue

        bbox_h = y2 - y1
        if bbox_h / frame_height < min_bbox_height_ratio:
            continue

        targets.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "class_name": cls_name,
            "confidence": float(box.conf[0]),
        })

    return targets


def draw_risk_zone(frame: np.ndarray, polygon: list, warning_active: bool) -> None:
    """
    在帧上绘制风险区域梯形。

    - 无预警：半透明黄色区域 + 黄色边界
    - 有预警：半透明红色区域 + 红色醒目边界
    """
    if warning_active:
        border_color = (0, 0, 255)      # red (BGR)
        fill_color = (0, 0, 255)
        fill_alpha = 0.15
        border_thickness = 3
    else:
        border_color = (0, 255, 255)    # yellow (BGR)
        fill_color = (0, 255, 255)
        fill_alpha = 0.10
        border_thickness = 2

    overlay = frame.copy()
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], fill_color)
    cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    cv2.polylines(frame, [pts], True, border_color, border_thickness)


def draw_warning_banner(frame: np.ndarray, text: str) -> None:
    """在画面顶部绘制红色预警横幅（文案由调用方决定）。"""
    h, w = frame.shape[:2]
    banner_h = int(h * 0.06)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font_scale = w / 1200.0
    thickness = max(2, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (banner_h + th) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)


def draw_risk_vehicle_boxes(frame: np.ndarray, risk_vehicles: list) -> None:
    """对每个风险车辆绘制加粗红色框 + RISK VEHICLE 标签。"""
    for v in risk_vehicles:
        x1, y1, x2, y2 = v["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        label = f"RISK VEHICLE ({v['class_name']})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1 - 2), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_risk_vru_boxes(frame: np.ndarray, risk_vrus: list) -> None:
    """对每个 VRU 风险目标绘制加粗品红色框 + RISK VRU 标签。"""
    for v in risk_vrus:
        x1, y1, x2, y2 = v["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
        label = f"RISK VRU ({v['class_name']})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1 - 2), (255, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
