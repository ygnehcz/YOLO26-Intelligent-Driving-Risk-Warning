# YOLO26 智能驾驶道路参与者检测与风险预警系统

基于 YOLO26 的智能驾驶道路目标检测与风险预警系统，支持对车载视频进行逐帧检测、车辆风险预警与弱势交通参与者（VRU）风险提示。

## 当前进度

- [x] 已完成 YOLO26 基础图片推理验证
- [x] 已完成道路视频逐帧推理工程脚本
- [x] 已生成首个智能驾驶道路检测结果视频
- [x] 已完成前方车辆风险区域绘制
- [x] 已完成基于图像空间区域的车辆风险预警提示
- [x] 已完成风险预警逻辑校准（增加近距视觉约束：bbox 高度比阈值）
- [x] 已完成弱势交通参与者（VRU）风险提示
- [x] 已完成 VRU 风险阈值校准（0.06 → 0.10，经实景视频验证）

## 项目结构

```
├── data/
│   └── test_videos/          # 测试视频（不入库）
├── scripts/
│   ├── predict_video.py      # 逐帧视频推理脚本（主入口）
│   ├── extract_frames.py     # 帧抽取工具
│   └── analyze_samples.py    # 抽样帧分析工具
├── utils/
│   ├── __init__.py
│   └── warning_utils.py      # 风险区域、双类预警判定、绘制工具
├── outputs/
│   └── videos/               # 检测结果视频（不入库）
├── .gitignore
└── README.md
```

## 快速开始

```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
.venv/Scripts/activate
pip install ultralytics opencv-python

# 运行视频检测
python scripts/predict_video.py
```

输出视频将保存在 `outputs/videos/` 目录下。

## 风险预警规则

### 两类风险目标

| 类别 | 关注目标 | bbox 高度比阈值 |
|------|----------|----------------|
| 车辆 (Vehicle) | car, bus, truck | >= 0.12 |
| 弱势交通参与者 (VRU) | person, bicycle, motorcycle | >= 0.10 |

> VRU 阈值 0.10 经 `road_city_vru_01.mp4` 城市行人/自行车实景视频校准，可过滤极远小目标，近处行人/骑行者仍正常触发。

### 判定规则

所有目标均需满足两层判定：
1. 检测框底边中心点落入前方风险梯形区域
2. 检测框高度 / 画面高度 >= 对应阈值

### 预警横幅优先级

| 条件 | 横幅文案 |
|------|----------|
| 车辆 + VRU 同时存在 | WARNING: VEHICLE AND VRU IN RISK ZONE |
| 仅车辆 | WARNING: VEHICLE IN RISK ZONE |
| 仅 VRU | WARNING: VRU IN RISK ZONE |
| 无风险 | 不显示 |

### 视觉区分

- 车辆风险框：红色 (Red) + `RISK VEHICLE` 标签
- VRU 风险框：品红色 (Magenta) + `RISK VRU` 标签
