# 项目开发记录

> 项目：基于 YOLO26 的智能驾驶道路参与者检测与风险预警系统
> 目录：D:\YOLO26-Intelligent-Driving-Risk-Warning

---

## 2026-05-15（第一天）

### 1. 项目立项与技术路线

- 最终选定 **YOLO26**（Ultralytics YOLO）作为道路目标检测底座
- 项目定位：智能驾驶道路参与者检测与风险预警系统
- 当前阶段策略：不进行复杂训练/微调，优先完成可运行的端到端工程系统
- 使用预训练权重 `yolo26n.pt`（nano 版本，适合 CPU 环境快速验证）

### 2. 环境搭建

| 组件 | 版本 |
|---|---|
| Python | 3.11.9 |
| ultralytics | 8.4.51 |
| torch | 2.12.0+cpu |
| OpenCV | (与 ultralytics 依赖一并安装) |
| 运行设备 | CPU（无 CUDA） |

- 已创建 Python 虚拟环境 `.venv/`
- `yolo checks` 已通过

### 3. 基础推理验证

#### 3.1 图片推理

```bash
yolo predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg
```

- 成功检测公交车场景，推理时间约 **49.7 ms**

#### 3.2 命令行视频推理

```bash
yolo predict model=yolo26n.pt source=data/test_videos/road_drive_01.mp4
```

- 成功运行道路视频检测，推理速度约 **28.1 ms/frame**
- 检测结果保存在 `runs/detect/predict-4/`

#### 3.3 测试素材

- 测试视频：`data/test_videos/road_drive_01.mp4`（1080×1920 / 30 FPS / 722 帧）
- 测试图片：`bus.jpg`

### 4. 阶段 2：工程化逐帧视频推理脚本

**目标**：从 `yolo predict` 一行式命令升级为可扩展的工程脚本

**新增文件**：
- `scripts/predict_video.py` — 核心脚本

**功能**：
- 使用 `ultralytics.YOLO` 类加载模型
- 逐帧读取 → 逐帧检测 → OpenCV 写入输出视频
- 保持原视频分辨率（1080×1920）和帧率（30 FPS）
- 中文终端日志：模型加载、视频信息、每 50 帧打印进度、最终输出路径
- 完善的异常处理：视频不存在/打不开均给出明确报错
- 支持命令行参数覆盖输入/输出路径
- `process_frame()` 预留扩展点，便于后续添加自定义风险逻辑

**输出视频**：`outputs/videos/road_drive_01_detected.mp4`

**Git 提交**：
```
c27f83c feat: add frame-by-frame YOLO26 video inference script
```
- 包含 `.gitignore`、`README.md`、`scripts/predict_video.py`（共 3 个文件）

### 5. 阶段 3：前方车辆风险区域与预警逻辑

**目标**：在逐帧检测基础上，增加前方风险区域绘制与车辆预警判定

**新增文件**：
- `utils/__init__.py` — Python 包初始化
- `utils/warning_utils.py` — 风险区域与预警逻辑模块

**修改文件**：
- `scripts/predict_video.py` — 集成风险预警流程
- `README.md` — 更新进度

#### 5.1 风险区域设计

- 使用**相对坐标**计算的梯形区域，自适应任意分辨率
- 梯形四点（相对于画面宽 W、高 H）：

| 顶点 | X 坐标 | Y 坐标 |
|---|---|---|
| 左上 (TL) | 0.38W | 0.58H |
| 右上 (TR) | 0.62W | 0.58H |
| 右下 (BR) | 0.92W | 0.98H |
| 左下 (BL) | 0.08W | 0.98H |

- 梯形位于画面底部中央，模拟本车前方道路区域

#### 5.2 车辆预警规则

- 关注类别：`car`、`bus`、`truck`、`motorcycle`
- 对每个目标框，取**框底边中心点** `((x1+x2)/2, y2)` 作为地面近似落点
- 使用射线法（Ray Casting）判断落点是否在风险梯形内
- 落点在区域内 → 该帧触发预警 `warning_active = True`

#### 5.3 视觉呈现

- **无预警帧**：半透明黄色梯形 + 黄色边界线
- **有预警帧**：
  - 风险区域变为半透明红色 + 红色醒目边界
  - 风险车辆绘制更粗的红色矩形框 + `RISK (class_name)` 标签
  - 画面顶部显示红色预警横幅：`WARNING: VEHICLE IN RISK ZONE`

#### 5.4 模块化设计

`utils/warning_utils.py` 提供的公共函数：

| 函数 | 职责 |
|---|---|
| `get_risk_zone_polygon(w, h)` | 返回自适应梯形四顶点 |
| `is_point_in_polygon(point, polygon)` | 射线法点-多边形判定 |
| `detect_risk_vehicles(result, class_names, polygon)` | 筛选落入风险区的车辆 |
| `draw_risk_zone(frame, polygon, warning_active)` | 绘制风险区域（两种模式） |
| `draw_warning_banner(frame)` | 绘制顶部红色预警横幅 |
| `draw_risk_vehicle_boxes(frame, risk_vehicles)` | 对风险车辆额外醒目标记 |

#### 5.5 运行结果

| 指标 | 数值 |
|---|---|
| 总处理帧数 | 722 |
| 风险预警帧数 | 722 |
| 风险预警帧占比 | 100.0% |

**输出视频**：`outputs/videos/road_drive_01_risk_warning.mp4`

**Git 提交**：
```
e1d6ee5 feat: add forward vehicle risk zone warning
```
- 包含 `utils/__init__.py`、`utils/warning_utils.py`、`scripts/predict_video.py`、`README.md`（共 4 个文件）

---

## 当前项目文件结构

```
D:\YOLO26-Intelligent-Driving-Risk-Warning\
├── .gitignore                         # Git 忽略规则
├── README.md                          # 项目说明与进度追踪
├── PROJECT_LOG.md                     # 本文档：阶段性开发记录
├── yolo26n.pt                         # YOLO26 nano 预训练权重（不入库）
├── bus.jpg                            # 测试图片（不入库）
│
├── scripts/
│   └── predict_video.py               # ★ 主脚本：逐帧检测 + 风险预警
│
├── utils/
│   ├── __init__.py
│   └── warning_utils.py               # 风险区域计算、车辆预警判定、绘制工具
│
├── data/
│   └── test_videos/
│       └── road_drive_01.mp4          # 测试视频（不入库）
│
├── outputs/
│   └── videos/                        # 输出视频目录（内容不入库）
│       ├── road_drive_01_detected.mp4
│       └── road_drive_01_risk_warning.mp4
│
├── runs/                              # ultralytics 自动生成的运行结果（不入库）
└── .venv/                             # Python 虚拟环境（不入库）
```

---

## 下一步建议

### 优先建议

1. **人工观看阶段 3 输出视频**
   - 打开 `outputs/videos/road_drive_01_risk_warning.mp4`
   - 判断风险区域是否过宽/过窄，预警是否过于频繁（当前 100% 预警率）

2. **根据效果微调风险区域坐标**
   - 修改 `utils/warning_utils.py` 中 `get_risk_zone_polygon()` 的相对坐标
   - 如果预警过密 → 缩小梯形范围（提高上边界 Y、收窄左右边界）
   - 如果预警遗漏 → 扩大梯形范围

3. **根据效果决定下一阶段方向**
   - 路线 A：增加行人/骑行者风险提示（扩展 `RISK_CLASS_NAMES` + 独立风险判定）
   - 路线 B：集成 ByteTrack 多目标跟踪，实现车辆 ID 稳定追踪
   - 暂不急于 GUI，也暂不急于训练数据集

---

## 明天从这里继续

### 第一步：观看输出视频

在文件资源管理器中打开：
```
outputs\videos\road_drive_01_risk_warning.mp4
```
观察：
- 梯形风险区域的位置是否合理（是否覆盖了前方车道区域）
- 预警横幅和 RISK 标记是否过于频繁
- 如果有漏检的前方车辆，记下大致帧号

### 第二步：查看核心脚本

```
scripts/predict_video.py     → 主流程入口
utils/warning_utils.py       → 风险区域参数和判定逻辑
```

### 第三步：根据效果分岔

| 观察结果 | 推荐操作 |
|---|---|
| 预警太多（100% 无间断） | 调整 `get_risk_zone_polygon` 坐标：缩小梯形上边宽度、提高上边界 Y |
| 预警偶尔遗漏 | 适当扩大梯形范围 |
| 效果满意 | 进入行人/骑行者风险提示 或 ByteTrack 多目标跟踪 |

### 关键可调参数

在 `utils/warning_utils.py` → `get_risk_zone_polygon()` 中（约第 18-21 行）：

```python
(int(0.38 * W), int(0.58 * H)),   # 左上 → 调大 X 或 Y = 缩小范围
(int(0.62 * W), int(0.58 * H)),   # 右上 → 调小 X 或 Y = 扩大范围
(int(0.92 * W), int(0.98 * H)),   # 右下
(int(0.08 * W), int(0.98 * H)),   # 左下
```

修改后直接运行 `python scripts/predict_video.py` 即可看到效果。

---

## 2026-05-16（第二天）：阶段 3.5 — 风险预警逻辑校准

### 1. 抽样分析

从原始视频与预警输出视频中，按固定间隔抽取 8 个关键帧（#1, #100, #200, #300, #400, #500, #600, #700）到 `outputs/analysis_samples/`。

抽样分析结论：
- **每个抽样帧都有 1 辆车落入风险梯形区域**（前车始终在画面中），这是 100% 预警率的直接原因
- 前车 bbox 高度比范围：0.105 ~ 0.312（帧 700 距离最近）
- 侧面车辆（低置信度、bbox 底边 y2≈H）被正确排除在梯形外，规则本身没有误判
- **判断**：100% 预警率源于视频内容（全程跟车），但"始终预警"削弱了预警的实际意义

### 2. 规则优化

采用**方案 A：目标框高度比例阈值**。

在 `utils/warning_utils.py` 中增加常量 `MIN_BBOX_HEIGHT_RATIO`：

```python
MIN_BBOX_HEIGHT_RATIO = 0.12
```

新增规则：检测框高度 / 画面高度 >= 0.12 才触发预警。

`detect_risk_vehicles()` 函数签名扩展：
- 新增 `frame_height` 参数（用于计算高度比）
- 新增 `min_bbox_height_ratio` 参数（可调整阈值）

终端日志增加规则描述：
```
Risk rule: point in polygon + bbox height ratio >= 0.12
```

### 3. 优化效果

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 总帧数 | 722 | 722 |
| 预警帧数 | 722 | 502 |
| 预警占比 | 100.0% | 69.5% |
| 减少不必要的预警 | — | 220 帧（30.5%） |

规则优点：
- 可解释性强：只有画面中足够大的前车才触发预警
- 阈值集中可调（`MIN_BBOX_HEIGHT_RATIO` 一处修改即生效）
- 保留了原梯形区域判定的全部逻辑，仅在其后追加轻量过滤

规则局限：
- 单帧独立判定，不考虑时序平滑（后续可引入跟踪或滑动窗口）
- 高度比阈值受摄像头 FOV 和分辨率影响，切换场景可能需要重新标定
- 未区分车辆类型（如卡车 vs 轿车在同一距离下高度比不同）

### 4. 修改文件

| 文件 | 变更 |
|---|---|
| `utils/warning_utils.py` | 新增 `MIN_BBOX_HEIGHT_RATIO` 常量；`detect_risk_vehicles()` 增加高度比约束 |
| `scripts/predict_video.py` | 传入 `frame_height`；新增规则日志；更新 docstring |
| `scripts/extract_frames.py` | 新增：帧抽取工具脚本 |
| `scripts/analyze_samples.py` | 新增：抽样帧分析脚本 |
| `README.md` | 更新进度与风险规则说明 |
| `PROJECT_LOG.md` | 追加本日开发记录 |

### 5. 输出

- 输出视频：`outputs/videos/road_drive_01_risk_warning.mp4`
- 抽样帧：`outputs/analysis_samples/`（16 张图片）

### 6. Git 提交

```
refine: calibrate forward vehicle risk warning rule
```

---

---

## 2026-05-16（续）：阶段 4 — 弱势交通参与者风险提示

### 1. 功能目标

在车辆风险预警基础上，新增对弱势交通参与者（VRU: Vulnerable Road User）的风险提示：
- person（行人）
- bicycle（骑行者）
- motorcycle（摩托车）

### 2. 实现方案

#### 2.1 风险类别划分

| 类别 | 常量 | 关注目标 | bbox 高度比阈值 |
|------|------|----------|----------------|
| 车辆 | `VEHICLE_CLASSES` | car, bus, truck | `MIN_VEHICLE_BBOX_HEIGHT_RATIO = 0.12` |
| VRU | `VRU_CLASSES` | person, bicycle, motorcycle | `MIN_VRU_BBOX_HEIGHT_RATIO = 0.06` |

说明：
- 车辆阈值 0.12 较严格，抑制远处小车
- VRU 阈值 0.06 较宽松（仅 6% 画面高度即触发），因为行人/骑行者目标尺寸天然较小

#### 2.2 预警横幅优先级

```
1. VEHICLE + VRU 同时存在 → WARNING: VEHICLE AND VRU IN RISK ZONE
2. 仅 VEHICLE            → WARNING: VEHICLE IN RISK ZONE
3. 仅 VRU                → WARNING: VRU IN RISK ZONE
4. 均无                  → 不显示横幅
```

横幅文案由 `build_banner_text(has_vehicle, has_vru)` 函数生成。

#### 2.3 视觉区分

| 风险类型 | 框颜色 | 标签 |
|----------|--------|------|
| 车辆 | 红色 Red (0,0,255) | `RISK VEHICLE (class_name)` |
| VRU | 品红 Magenta (255,0,255) | `RISK VRU (class_name)` |

新增函数：
- `draw_risk_vru_boxes(frame, risk_vrus)` — 品红色加粗框 + VRU 标签
- `build_banner_text(has_vehicle, has_vru)` — 横幅文案优先级逻辑

#### 2.4 代码重构

| 旧名称 | 新名称 | 说明 |
|--------|--------|------|
| `RISK_CLASS_NAMES` | `VEHICLE_CLASSES` + `VRU_CLASSES` | 拆分为两组常量 |
| `MIN_BBOX_HEIGHT_RATIO` | `MIN_VEHICLE_BBOX_HEIGHT_RATIO` + `MIN_VRU_BBOX_HEIGHT_RATIO` | 独立阈值 |
| `detect_risk_vehicles(result, class_names, ...)` | `detect_risk_targets(result, class_names, ...)` | 通用化，参数化 |
| `draw_warning_banner(frame)` | `draw_warning_banner(frame, text)` | 文案由调用方传入 |

`process_frame()` 返回值从 `(frame, result, warning_active)` 改为 `(frame, has_vehicle, has_vru)`。

### 3. 运行统计

| 指标 | 数值 |
|------|------|
| 总处理帧数 | 722 |
| 车辆风险预警帧数 | 502 |
| VRU 风险预警帧数 | 0 |
| 车辆与 VRU 同时预警 | 0 |
| 任一风险预警帧数 | 502 |
| 任一风险预警占比 | 69.5% |

VRU 预警帧数为 0 的原因：当前测试视频为高速/快速路场景，未见行人/骑行者出现。车辆预警统计与阶段 3.5 完全一致，确认无回归。

### 4. 修改文件

| 文件 | 变更 |
|------|------|
| `utils/warning_utils.py` | 拆分 VEHICLE/VRU 类别与阈值；`detect_risk_targets()` 通用化；新增 `draw_risk_vru_boxes()` |
| `scripts/predict_video.py` | 双类检测；横幅优先级逻辑；新统计输出；默认输出路径更新 |
| `README.md` | 更新进度、功能说明、规则表格 |
| `PROJECT_LOG.md` | 追加本阶段记录 |

### 5. 输出视频

`outputs/videos/road_drive_01_multi_risk_warning.mp4`

（阶段 3.5 输出视频 `road_drive_01_risk_warning.mp4` 保留不覆盖）

### 6. Git 提交

```
feat: add vulnerable road user risk warning
```

---

## 下一步建议

1. 观看新输出视频，确认 VRU 视觉标记效果
2. 如需测试 VRU 功能，需准备包含行人/骑行者的城市道路视频
3. 后续可考虑引入 ByteTrack 多目标跟踪，实现稳定的车辆/VRU ID 追踪与 TTC 估计
