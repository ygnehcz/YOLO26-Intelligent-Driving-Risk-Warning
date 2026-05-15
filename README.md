# YOLO26 智能驾驶道路参与者检测与风险预警系统

基于 YOLO26 的智能驾驶道路目标检测与风险预警系统，支持对车载视频进行逐帧检测并预留风险预警扩展能力。

## 当前进度

- [x] 已完成 YOLO26 基础图片推理验证
- [x] 已完成道路视频逐帧推理工程脚本
- [x] 已生成首个智能驾驶道路检测结果视频
- [x] 已完成前方车辆风险区域绘制
- [x] 已完成基于图像空间区域的车辆风险预警提示
- [x] 已完成风险预警逻辑校准（增加近距视觉约束：bbox 高度比阈值）

## 项目结构

```
├── data/
│   └── test_videos/          # 测试视频（不入库）
├── scripts/
│   └── predict_video.py      # 逐帧视频推理脚本
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

- **梯形区域判定**：检测框底边中心点是否落入前方风险梯形区域
- **近距视觉约束**：检测框高度 / 画面高度 >= 0.12（抑制远处小车误触发）
- 规则已由"仅落点判定"升级为"落点 + 近距约束"
