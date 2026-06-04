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

## 2026-05-16 凌晨收工记录

### 今天实际完成的内容

#### 阶段 3.5：前方车辆风险规则校准

- 从单纯"检测框底边中心点进入梯形区域"升级为：
  **point in polygon + bbox height ratio >= 0.12**
- 预警帧数从 722/722（100%）降为 502/722（69.5%）
- 新增抽样分析工具脚本：`extract_frames.py`、`analyze_samples.py`
- 对应 commit：`835d523 refine: calibrate forward vehicle risk warning rule`

#### 项目首次推送到 GitHub

- 远程仓库：https://github.com/ygnehcz/YOLO26-Intelligent-Driving-Risk-Warning
- 本地分支 `master` → 重命名为 `main`
- `main` 正确跟踪 `origin/main`
- 推送成功后可见全部已入库文件

#### 阶段 4：新增 VRU 风险提示

- 拆分风险类别：
  - `VEHICLE_CLASSES = {car, bus, truck}` → 阈值 0.12
  - `VRU_CLASSES = {person, bicycle, motorcycle}` → 阈值 0.06
- 代码重构：
  - `detect_risk_vehicles()` → `detect_risk_targets()`（通用化）
  - `draw_warning_banner()` 改为参数化文案
  - 新增 `draw_risk_vru_boxes()`（品红色标记）
  - 新增 `build_banner_text()` 横幅优先级逻辑
- 横幅优先级：车辆+VRU > 仅车辆 > 仅 VRU > 无
- 输出视频：`outputs/videos/road_drive_01_multi_risk_warning.mp4`
- 运行统计：
  - 总帧数 722
  - 车辆风险预警 502 帧
  - VRU 风险预警 0 帧
  - 同时预警 0 帧
  - 任一风险预警 502 帧（69.5%）
- 对应 commit：`eac70f5 feat: add vulnerable road user risk warning`

### 当前阶段判断

VRU 逻辑代码已完整接入，但**当前测试视频为高速/快速路场景，无行人、骑行者或摩托车出现**，VRU 功能尚未通过真实含 VRU 的视频验证。这是明天优先要补的验证项。

### 当前 Git 状态

- 分支：`main`，与 `origin/main` 同步
- 工作区：干净
- 最新 commit：`eac70f5 feat: add vulnerable road user risk warning`

---

## 明天从这里继续

1. **优先寻找一段包含行人 / 自行车 / 摩托车的城市道路视频**；放在 `data/test_videos/` 目录下
2. 使用当前脚本 `python scripts/predict_video.py`（可命令行指定输入视频路径）跑通 VRU 风险提示
3. 检查 VRU 阈值 0.06 是否过松（远处小人误触发）或过严（较近行人也未触发）
4. 如果 VRU 验证合理，再决定进入：
   - ByteTrack 多目标跟踪（车辆/VRU ID 稳定追踪 + TTC 估计）
   - 或先做预警时序稳定性优化（滑动窗口 / 迟滞阈值，减少临界距离处预警闪烁）
5. 如需，可准备多段不同场景视频，逐步构建项目验证集

---

## 2026-06-04：阶段 4.5 — VRU 风险提示验证

### 1. 测试视频

- 来源：YouTube 公开视频（ID `6mATXRigCJU`），城市街道行人/自行车场景
- 保存为：`data/test_videos/road_city_vru_01.mp4`
- 规格：1280×720 / 30 FPS / 1800 帧 / 60 秒

### 2. 验证结果（阈值 0.06）

| 指标 | 数值 |
|------|------|
| 总帧数 | 1800 |
| 车辆风险预警 | 0 |
| VRU 风险预警 | 1732 |
| 同时预警 | 0 |
| 任一风险预警 | 1732（96.2%） |

- **RISK VRU 品红标记已确认出现**，VRU 横幅文案路径正常
- `motorcycle` 有检出（2 次），归类到 VRU 正确
- 视频无车辆，车辆+VRU 同时预警路径未验证

### 3. VRU 阈值分析

20 帧采样（244 个 VRU 检测框）高度比分布：

| 区间 | 数量 | 占比 |
|------|------|------|
| [0.00, 0.06) | 1 | 0.4% |
| [0.06, 0.10) | 12 | 4.9% |
| [0.10, 1.00) | 231 | 94.7% |

- 99.6% 的检测框 ≥ 0.06，阈值几乎没有过滤效果
- 中位数 0.33，均值 0.35——视频中行人目标天然很大

---

## 2026-06-04（续）：阶段 4.6 — VRU 阈值第一轮校准

### 1. 修改内容

`utils/warning_utils.py` — `MIN_VRU_BBOX_HEIGHT_RATIO`：**0.06 → 0.10**

### 2. 新旧对比

| 指标 | 旧阈值 0.06 | 新阈值 0.10 | 变化 |
|------|------------|------------|------|
| 总帧数 | 1800 | 1800 | — |
| 车辆预警 | 0 | 0 | — |
| VRU 预警 | 1732 | 1730 | −2 |
| 同时预警 | 0 | 0 | — |
| 任一预警 | 1732 | 1730 | −2 |
| 预警占比 | 96.2% | 96.1% | −0.1% |

### 3. 差异帧

仅 2 帧从预警变为非预警：

| 帧号 | 目标 | 高度比 |
|------|------|--------|
| 1127 | person | 0.0931 |
| 1139 | person | 0.0972 |

均为画面远处的极小行人目标，属于边缘案例。

### 4. 结论

- **0.10 可过滤极远小目标**（高度比 < 0.10），近处行人/自行车仍正常触发
- 当前高预警率（96.1%）主要由视频中密集 VRU 场景导致，而非代码错误——绝大多数行人在风险梯形区域内
- 后续如需进一步降低误报，应从**风险梯形区域形状**入手，而非继续上调阈值

### 5. 输出视频

- `outputs/videos/road_city_vru_01_multi_risk_warning.mp4`（阈值 0.06，保留）
- `outputs/videos/road_city_vru_01_multi_risk_warning_vru010.mp4`（阈值 0.10，当前正式输出）

### 6. 修改文件

| 文件 | 变更 |
|------|------|
| `utils/warning_utils.py` | `MIN_VRU_BBOX_HEIGHT_RATIO` 0.06 → 0.10 |
| `README.md` | 更新 VRU 阈值 + 校准说明 |
| `PROJECT_LOG.md` | 追加阶段 4.5 + 4.6 记录 |

### 7. Git 提交

```
refine: calibrate VRU risk warning threshold
```

---

## 明天从这里继续

1. **当前高预警率瓶颈在风险梯形区域形状**，而非阈值。考虑：
   - 缩窄梯形上边宽度（减小 0.38W~0.62W 的范围）
   - 或提高梯形上边界 Y（从 0.58H 提高）
2. **寻找同时含车辆 + VRU 的视频**，验证"车辆+VRU 同时预警"横幅路径
3. 如果 VRU 验证整体满意，可进入：
   - ByteTrack 多目标跟踪（车辆/VRU ID 稳定追踪 + TTC 估计）
   - 或预警时序稳定性优化（滑动窗口 / 迟滞阈值，减少临界帧闪烁）
4. `road_city_vru_01.mp4` 已作为项目 VRU 验证视频入库

---

## 2026-06-04（续）：阶段 4.7 — 测试样例说明整理

### 1. 当前测试样例

项目已形成两个代表性测试样例，覆盖车辆风险与 VRU 风险两个维度：

| 视频 | 场景 | 验证目标 | 输出 |
|------|------|----------|------|
| `road_drive_01.mp4` | 快速路 / 跟车 | 车辆风险预警 | `road_drive_01_multi_risk_warning.mp4` |
| `road_city_vru_01.mp4` | 城市道路 / 行人自行车 | VRU 风险提示 | `road_city_vru_01_multi_risk_warning_vru010.mp4` |

### 2. 修改文件

| 文件 | 变更 |
|------|------|
| `README.md` | 新增 Test Videos 小节 |
| `PROJECT_LOG.md` | 追加阶段 4.7 记录 |

### 3. Git 提交

```
docs: describe test videos and validation scenarios
```

---

## 明天从这里继续

1. 后续 ByteTrack 多目标跟踪阶段应优先基于这两个视频继续测试
2. **当前高预警率瓶颈在风险梯形区域形状**，而非阈值。考虑：
   - 缩窄梯形上边宽度（减小 0.38W~0.62W 的范围）
   - 或提高梯形上边界 Y（从 0.58H 提高）
3. **寻找同时含车辆 + VRU 的视频**，验证"车辆+VRU 同时预警"横幅路径
4. 如果预警逻辑整体满意，可进入：
   - ByteTrack 多目标跟踪（车辆/VRU ID 稳定追踪 + TTC 估计）
   - 或预警时序稳定性优化（滑动窗口 / 迟滞阈值，减少临界帧闪烁）
5. 测试视频和输出视频均不进入版本控制（`.gitignore` 已配置）

---

## 2026-06-04（续）：阶段 5 — ByteTrack 多目标跟踪与目标 ID 标注

### 1. 功能目标

在 YOLO26 检测 + 风险预警基础上，集成 ByteTrack 多目标跟踪：
- 给每个目标（车辆/行人/自行车/摩托车）分配稳定 Track ID
- 输出视频中显示 ID
- 统计每个风险目标的持续帧数
- 为后续时序预警平滑和持续风险判定打基础

### 2. 实现方案

#### 2.1 新增脚本

`scripts/track_video.py` — 基于 `model.track()` 的独立跟踪脚本。

- 使用 `model.track(frame, persist=True, tracker="bytetrack.yaml")`
- 支持 `--model` / `--input` / `--output` / `--tracker` 参数
- 保持原视频分辨率、FPS、总帧数

#### 2.2 视觉层次

| 类型 | 框颜色 | 标签 |
|------|--------|------|
| 普通目标 | 绿色 | `class_name ID:#` |
| 风险车辆 | 红色粗框 | `RISK VEHICLE (class_name)` |
| 风险 VRU | 品红粗框 | `RISK VRU (class_name)` |

风险目标跳过绿色框，由专用函数绘制醒目标记。

#### 2.3 代码复用

- 风险判定逻辑完全复用 `utils/warning_utils.py`（`detect_risk_targets` 等）
- `detect_risk_targets` 返回值新增可选 `track_id` 字段，兼容旧调用方
- `draw_risk_zone` / `draw_warning_banner` / `draw_risk_vehicle_boxes` / `draw_risk_vru_boxes` 全部复用

### 3. 运行统计

#### road_drive_01.mp4（车辆场景）

| 指标 | 数值 |
|------|------|
| 总帧数 | 722 |
| 唯一 Track ID 数 | 1 |
| 车辆风险预警帧数 | 502 |
| VRU 风险预警帧数 | 0 |
| 任一风险预警占比 | 69.5% |
| 风险车辆 | ID 1: 502 帧 |

与 predict_video.py 结果完全一致，确认无回归。

#### road_city_vru_01.mp4（VRU 场景）

| 指标 | 数值 |
|------|------|
| 总帧数 | 1800 |
| 唯一 Track ID 数 | 316 |
| 车辆风险预警帧数 | 0 |
| VRU 风险预警帧数 | 1724 |
| 任一风险预警占比 | 95.8% |

VRU 风险目标中，持续帧数突出的 ID：
- ID 2253: 290 帧（出现最长）
- ID 618: 256 帧
- ID 1801: 248 帧
- ID 232: 175 帧
- ID 2667: 172 帧

VRU 预警帧 1724 vs predict 模式 1730（差 6 帧，0.3%），差异来自 tracking 模式对检测框的微小变化。

### 4. 输出视频

| 视频 | 输出路径 |
|------|----------|
| road_drive_01 | `outputs/videos/road_drive_01_tracked.mp4` |
| road_city_vru_01 | `outputs/videos/road_city_vru_01_tracked.mp4` |

### 5. 修改文件

| 文件 | 变更 |
|------|------|
| `scripts/track_video.py` | 新增：ByteTrack 多目标跟踪脚本 |
| `utils/warning_utils.py` | `detect_risk_targets` 返回值新增可选 `track_id` 字段 |
| `README.md` | 新增多目标跟踪章节 + 跟踪统计表 + 进度更新 |
| `PROJECT_LOG.md` | 追加阶段 5 记录 |

### 6. Git 提交

```
feat: add ByteTrack multi-object tracking
```

### 7. 当前局限

- 不绘制轨迹线（当前只显示 ID）
- VRU 场景中 ID 较多（316 个），部分 ID 仅出现 1-2 帧（ByteTrack ID 碎片化）
- 行人密集场景下 ID Switch 较频繁，稳定跟踪长距离行人仍有挑战
- 未区分 risk persistence（持续风险时长），仅做了帧数统计

---

## 明天从这里继续

1. **利用 Track ID 实现时序平滑**：
   - 对风险目标设置最小持续帧数阈值（如 ≥ 5 帧才触发预警），减少瞬间误报
   - 或使用滑动窗口 / 迟滞阈值减少临界帧处的预警闪烁
2. **风险梯形区域重设计**（当前高预警率的核心瓶颈）
3. **寻找同时含车辆 + VRU 的视频**，验证"车辆+VRU 同时预警"横幅路径
4. 后续可增加轨迹线绘制、TTC 估计等高级功能
