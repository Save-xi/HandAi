# 单右手视觉遥操作与意图预测阶段完成说明书

首次成稿：2026-08-24；Phase 1.5 二轮风险加固更新：2026-08-29
Python 工程：`D:\VR\HandAi\single-hand-teleop-baseline`
Unity 工程：`D:\SVH\RoboticArm`
Phase 1 clean-run 代码基线：`a5a53da`（PR #14 已合并）；Phase 1.5 为当前工作树更新

> 本说明书回答三件事：我们实际做了什么、结果达到什么程度、接下来最值得做什么。
> 所有“完成”都按证据范围表述。当前主线始终是**单右手 + Unity 虚拟预览**，不是双手，也不是真机控制。

> 2026-08-29 当前口径：open-release 改变了 H2O 代理标签语义，旧 v1 checkpoint 已由映射契约闸门淘汰。
> 当前 v2 重训结果改善但未通过全部预注册门槛，只能保留为默认关闭的影子诊断。本文后面的 v1
> 3.60%/5.73% 表格只作历史实验记录；现役证据以
> [Phase 1.5 二轮风险加固完成记录](phase15_risk_hardening.md) 为准。

## 1. 一页结论

当前已经形成一条可运行、可测试、可离线验收的单右手主链：

```text
摄像头 / 视频
  -> MediaPipe 21 点手部检测
  -> 选择置信度最高的 Right 手
  -> 连续几何特征 + 质量门控 + 基础手势
  -> control_representation（硬件无关控制中间层）
  -> svh_preview（9 通道归一化预览目标）
  -> JSON / JSONL / 可选本机 UDP
  -> Unity 虚拟手关节展开

当前帧 UDP 发送后
  -> 非阻塞提交到默认关闭的 residual GRU worker
  -> 30 Hz 重采样
  -> hold/raw/gated 只写独立本机 prediction 日志
```

在此基础上，又完成了一条彼此隔离的 AI 研究支线：

```text
过去 30 帧单右手 9 通道目标
  -> hold-last / Linear / Kalman / GRU / TCN / Transformer 对照
  -> 零残差初始化 residual GRU
  -> 只用 validation 拟合的运动门控
  -> 预测未来 50 / 100 / 150 ms 的 9 通道目标
```

截至本文档日期，各部分状态如下。

| 部分 | 当前状态 | 最强证据 | 结论边界 |
|---|---|---|---|
| 单右手视觉主链 | 已实现 | 当前现役环境全量回归 `168 passed`；摄像头 60 帧无界面短跑 | 短跑时右手未入镜，仍需人工动作集验收 |
| payload / JSON / JSONL / UDP contract | 已实现并冻结 v1 行为 | schema、规范化/校验、输出顺序和回归测试 | UDP 是本机预览通道，不是可靠控制总线 |
| Phase 1 FreiHAND 离线验收 | 已通过，可冻结 | clean run `20260824T022030_928281Z_a5a53da_clean`，`release_eligible=true` | 静态图像，不等于实时视频或右手筛选验收 |
| Unity 新编辑器兼容与 UDP 安全 | 已完成自动行为回归 | Unity 2020.3.49f1 编译返回 0；loopback 有效/invalid/乱序/过期/watchdog batch PASS | 自动回归不替代正式场景视觉观感验收 |
| H2O 数据预处理 | v2 已完成且不覆盖 v1 | 113,322 有效帧、217 连续序列、mapping contract SHA 冻结 | 9 通道是姿态映射出的代理目标，不是真机传感器真值 |
| 第一轮意图预测 | 已完成 | 六类方法统一比较、正式报告和机器可读快照 | 深度模型未在整体 MAE 上击败 hold-last |
| 第二轮意图预测 | v1 历史 6/6；现役 v2 为 4/6、gate 未通过 | v2 总体 RMSE 改善 2.53%，q90 RMSE 改善 3.75%，P95 改善 5.59% | 只允许影子学习，不能宣称延迟补偿有效 |
| 默认关闭的预测影子模式 | 已实现非阻塞 v2 | mapping/SHA 闸门、30 Hz 重采样、独立 prediction 日志、真实日志 458 predicted | 不改 Unity/UDP；模型质量 gate 未通过 |
| 双手、多手 | 未做，且当前不要求 | — | 不应从 `max_num_hands=2` 推断项目是双手链路 |
| 实体 SVH / AUBO / 5G | 未接入 | — | 不具备真机协议、限位、ACK、watchdog、急停等安全证据 |

一句话概括：**单右手 Unity 虚拟预览的软件安全闭环更扎实了；延迟注入已决定控制参考继续使用 hold-last，预测分支如实停留在影子层，而不是急着扩双手或上真机。**

## 2. 项目边界与正确理解

### 2.1 当前真正完成的对象

当前系统控制的是 Unity 中的虚拟手预览。Python 输出的 9 通道含义是：

1. `thumb_flexion`
2. `thumb_opposition`
3. `index_finger_distal`
4. `index_finger_proximal`
5. `middle_finger_distal`
6. `middle_finger_proximal`
7. `ring_finger`
8. `pinky`
9. `finger_spread`

视觉关键点与 Unity 关节不是一一对应关系。正确的数据关系是：

```text
21 个视觉关键点
  -> 捏合距离、张开度、五指弯曲度等连续特征
  -> grasp / pinch 等硬件无关控制语义
  -> 9 个 SVH preview 通道
  -> Unity 中约 20 个虚拟关节的展开与耦合
```

因此，中指、无名指和小指不能完全像真人一样独立，并不自动说明视觉链坏了；机械手自由度、关节耦合以及 9 通道到虚拟关节的展开都会造成差异。

### 2.2 当前明确没有完成的对象

- 没有完成双手、多手或 InterHand2.6M 主线。
- 没有完成实体 SVH 的安全控制。
- 没有完成 TCP、串口或 RS485 真机传输。
- 没有完成 HoloLens、Kinect、5G 或 VR 头显端部署。
- 没有证明摄像头快速运动、遮挡、复杂背景和无手误检性能。
- 没有证明从相机曝光到 Unity 渲染完成或机械手动作完成的端到端时延。

## 3. 我们具体做了什么

### 3.1 恢复并固定单右手主线

首先回看了项目 README、Unity 交接、UDP 配置和源码，确认现有成果不是旧的三手势 demo，而是连续控制链。随后把工程约束写入 `AGENTS.md` 和各层文档：

- 主线只选择置信度最高的 `Right` 手；
- Unity 是可选下游，不是 Python baseline 的启动前提；
- 真机 SVH、串口服务和网络链路不是当前 baseline 的依赖；
- 新能力优先复用既有 JSON/JSONL contract，不另造不兼容接口。

配置中的 `max_num_hands: 2` 只表示 MediaPipe 最多产生两个候选，以便从中筛出右手；主循环仍只消费一只右手。

### 3.2 清理 Phase 1 的接口与运行风险

对主链做了以下安全性和可复现性处理：

- 把摄像头和视频文件收敛到统一 `InputSource` 接口，便于无摄像头时用固定视频回放。
- 对检测结果执行右手筛选，且只选置信度最高的一只 `Right`。
- 增加 landmark 质量门控，拦截掌心贴边或大量越界但检测器仍返回结果的高风险帧。
- 明确 `detected=true` 时 2D/3D landmark 都必须恰好为 21 点；`detected=false` 时两组点必须为空。
- 即使未检测到手、扩展关闭或扩展异常，`control_representation` 和 `svh_preview` 仍保留规范的 invalid 占位对象，避免下游因字段突然消失而崩溃。
- 所有数值在导出前统一 normalize + validate；schema、示例 payload 和测试同步维护。
- Unity UDP 默认关闭；只有 `configs/unity_udp_preview.yaml` 或明确的发送参数才会打开。
- 输出实时优先级固定为 **UDP -> 控制台 -> JSON/JSONL 落盘**，避免调试 I/O 延迟 Unity 预览。
- last-frame JSON 支持节流写盘，JSONL 支持批量 flush；单一 I/O 通道失败后安全停用该通道，不反复刷错拖垮主循环。
- 新增 `timing v1`：记录输入读取、检测、baseline、preview、payload ready 和 UDP 发送尝试时刻。

`timing v1` 能诊断 Python 阶段、同机 UDP 交付和 Unity 主线程排队，但不等于显示器真正完成渲染，更不等于实体机械手响应时间。

### 3.3 构建可重复的 Phase 1 离线验收

新增 `run_phase1_offline_acceptance.py`，把原来分散的检查收敛为一次不可覆盖的 run：

```text
创建唯一 run 目录
  -> 记录 Git commit / branch / dirty 状态
  -> 冻结有效配置与源码 SHA-256
  -> compileall
  -> pytest
  -> 在 FreiHAND evaluation 全量生成当前 pipeline 预测
  -> 对齐 GT 并计算指标
  -> 生成 Markdown / SVG / PPT 表格
  -> 为产物生成 SHA-256
  -> 判断 release_eligible
```

2026-08-24 在干净提交 `a5a53da` 上重新运行，得到：

```text
D:\VR\HandAi\single-hand-teleop-baseline\experiments\freihand_eval\outputs\phase1_acceptance\
  20260824T022030_928281Z_a5a53da_clean\
```

该 run 状态为 `completed`，8 项自动门槛全部通过，`release_eligible=true`。这消除了旧正式 run 因 dirty 工作树而不能作为冻结证据的问题。

### 3.4 增加无摄像头 Unity 合成验收

新增 `scripts/send_unity_preview_demo.py`：

- 不调用摄像头；
- 不调用 MediaPipe；
- 生成符合 payload contract 的 21 点占位和连续 9 通道目标；
- 默认只生成并校验，不发送网络；
- 加 `--send` 后只向本机 `127.0.0.1:18080` 发送；
- 动作序列固定为 `open -> pinch -> open -> fist -> open`；
- 默认拒绝远程地址，避免误把预览数据发到外部设备。

它用于证明 Python payload、UDP、本机 Unity 接收器和虚拟手展开能共同工作；它不验证视觉算法。

### 3.5 修复新版 Unity 打开工程的红色错误

实际启动版本是 Unity `2020.3.49f1`，旧工程还残留该版本不存在的内置包：

```text
com.unity.modules.autostreaming@1.0.0
```

完成的修复与验证包括：

- 从 `D:\SVH\RoboticArm\Packages\manifest.json` 删除失效依赖；
- 从 `D:\SVH\RoboticArm\Packages\packages-lock.json` 删除对应锁定项；
- 未发现真实 `CSxxxx` 源码编译错误；旧 DLL API 更新错误在完整重导入后未复现；
- 用 Unity 自身执行批处理重新导入、包解析和 C# 编译，返回码为 0；
- 打开 Editor 实际复核，非运行态 Console 为 0 日志、0 警告、0 错误；
- Play Mode 中监听 `127.0.0.1:18080`，发送合成单右手 9 通道包后虚拟手明显弯曲；
- 运行态 Console 保持 1 条正常监听日志、0 警告、0 错误；
- `applyBaselinePreviewToHardware=false` 全程保持关闭。

Editor 顶部的黄色 `Preview Packages in Use` 来自 URDF preview 包，是提示而不是红色编译错误；当前没有为了消除黄色提示而冒险升级机器人依赖。

### 3.6 固化 Python -> Unity 的预览接收逻辑

`RobotControlScript.cs` 当前完成了：

- UDP 端口 `18080` 的后台监听；
- 用线程安全队列把网络接收与 Unity 主线程分离；
- 丢弃已经过时的排队包，只在主线程应用最新包，降低积压；
- 优先读取合法的 `svh_preview.target_positions`；
- 9 通道到虚拟手关节的展开和 middle/ring/little 轻微显示增益；
- 可选的连续特征兜底路径；
- `timing v1` 的 Python、UDP、Unity queue、source-to-target-apply 诊断；
- 覆盖包和 frame gap 计数；
- `OnDestroy` 时关闭 UDP socket、线程和已有设备对象。

默认安全开关是：

```text
enableBaselineUdpPreview       = true
baselineUdpListenPort          = 18080
applyBaselinePreviewToHardware = false
logBaselinePreviewPackets      = false
enableLegacyGestureSnapping    = false
```

### 3.7 引入 H2O 公共数据，不做逐帧手工标注

使用 H2O 的四名受试者 pose-only 文件。每帧已有左右手有效位和 `21 x 3` 姿态，因此不需要人工逐帧画关键点。

转换过程为：

```text
读取右手 21 x 3 pose
  -> 腕点为原点、腕到中指 MCP 为单位掌长的标准化
  -> 复用 hand_features
  -> 复用 control_representation
  -> 复用 svh_adapter
  -> 保存连续 9 通道 NPZ + manifest + 哈希
```

数据处理结果：

| 项目 | 数量 |
|---|---:|
| take | 184 |
| pose 文件 | 114,329 |
| 无效帧 | 933 |
| 解析拒绝帧 | 0 |
| 最终连续序列 | 217 |
| 最终有效帧 | 113,322 |

严格跨受试者切分如下：

| split | subject | 序列 | 帧 | 窗口 |
|---|---|---:|---:|---:|
| train | subject1 + subject2 | 105 | 54,243 | 25,364 |
| validation | subject3 | 60 | 28,631 | 13,310 |
| test | subject4 | 52 | 30,448 | 14,352 |

三组 `sequence_id` 交集为 0；窗口不会跨越无效帧或序列边界。H2O 原始数据、处理数据和 checkpoint 都不提交到 Git。

### 3.8 完成第一轮统一模型比较

统一输入为过去 30 帧 9 通道，统一输出为未来 50/100/150 ms 的 9 通道目标。第一轮正式结果为：

| 模型 | 参数量 | 模型阶段耗时 | 单窗 P50 | MAE | RMSE | P95 | RMSE 相对 hold 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hold-last | 0 | — | — | **0.019952** | 0.055138 | 0.075735 | 0.00% |
| Linear | 0 | — | — | 0.026122 | 0.066519 | 0.109300 | -20.64% |
| Kalman-CV | 0 | — | — | 0.027071 | 0.069681 | 0.115393 | -26.38% |
| GRU | 172,699 | 16.77 s | **0.60 ms** | 0.021956 | 0.052936 | 0.074328 | +3.99% |
| TCN | 186,555 | 5,891.45 s | 7.65 ms | 0.021693 | **0.052681** | **0.073839** | **+4.46%** |
| Transformer（严格确定性） | 422,811 | 94.35 s | 15.40 ms | 0.022543 | 0.053818 | 0.076467 | +2.39% |

第一轮最重要的结论不是“深度模型全面胜利”，而是：

- H2O 有大量低运动窗口，短预测距离下 hold-last 是很强的 baseline；
- GRU/TCN 主要减少少数大误差，因此 RMSE 和高运动尾部改善，但整体 MAE 反而略差；
- Linear/Kalman 会把 9 通道抖动当成速度外推，容易过冲；
- TCN 只比 GRU多一点精度，却在严格确定性设置下耗时约 98 分钟，不适合作为当前本科项目的优先路线；
- GRU 更轻、更快、机制容易解释，适合作为第二轮基础。

### 3.9 历史 v1：第二轮 residual GRU + validation 门控

第二轮没有直接让神经网络从零预测绝对目标，而是做了三层保护：

1. 以最后一帧 hold-last 为基准，只预测有界残差；
2. 输出层零初始化，训练前严格等于 hold-last；
3. 门控只用已观测历史运动强度和 validation 拟合，并允许 `alpha=0` 精确退回 hold-last。

程序顺序固定为：

```text
只加载 train + validation
  -> 训练候选并拟合 validation 门控
  -> 写 selection.json 并计算 SHA-256
  -> 之后才加载 subject4 test
  -> 只评估 hold-last 和已冻结的选中候选
```

选中方案为 `residual_unweighted`，门控参数：最近 8 帧、validation 第 25 百分位运动阈值、三个预测距离均 `alpha=0.75`。`selection.json` SHA-256 为：

```text
9cc711e18d911907c8e678bb984f97f2318dd66484079844494b1c832014a231
```

subject4 结果：

| 方法 | MAE | RMSE | P95 绝对误差 | 越界率 |
|---|---:|---:|---:|---:|
| hold-last | 0.01995203 | 0.05513830 | 0.07573530 | 0 |
| residual GRU + frozen gate | **0.01934165** | **0.05315119** | **0.07149785** | 0 |
| 相对改善 | **3.06%** | **3.60%** | **5.60%** | — |

validation-q90 高运动子集上，RMSE 从 `0.12035698` 降至 `0.11345640`，改善 `5.73%`；P95 从 `0.32472156` 降至 `0.30297103`，改善 `6.70%`。

模型参数量 172,699，checkpoint 约 681 KiB；RTX 4060 Laptop 单窗口推理 P50/P95/最大值为 `1.028/1.503/1.812 ms`。预先写入配置的 6 项离线准入条件全部通过。

必须保留两条结果纪律：

- raw test RMSE `0.05259274` 虽比门控结果更好，但 raw/gated 二次选择没有在 test 前预注册，所以没有事后改正式方案；
- subject4 已被第一轮看过，第二轮属于内部复验，不是整个项目历史上从未见过的最终盲测。

### 3.10 整理文档、Git 与 CI

完成了以下研究与工程记录：

- Phase 1 无摄像头验收说明；
- Unity 重开与联动交接；
- 下游 preview contract；
- 意图预测第一轮、第二轮中文报告；
- AI 部分与计划书口径对齐说明；
- 正式 JSON/CSV 报告快照；
- Windows + Ubuntu GitHub Actions 回归。

Git 整理结果：

- 删除 Python/pytest 缓存、空临时文件、中断实验和错误 seed 的诊断 run；
- 保留三组正式意图预测报告，不提交原始数据与 checkpoint；
- 根 `.gitignore` 排除 `/datasets/` 和 `项目.pdf`；
- 子项目 `.gitignore` 排除运行输出、临时文件和缓存；
- 提交 `41018f2 feat: add offline acceptance and intent prediction`；
- 提交 `a5a53da ci: enforce UTF-8 on Windows runners`；
- [PR #14](https://github.com/Save-xi/HandAi/pull/14) `feat: add phase1 offline acceptance and intent prediction` 已合并；
- GitHub Actions 的 Windows 与 Ubuntu job 均为 success。

## 4. 正式结果汇总

### 4.1 Phase 1 clean run

正式 run：`20260824T022030_928281Z_a5a53da_clean`

| 指标 | 阶段门槛 | 实测 | 结果 |
|---|---:|---:|---|
| GT / prediction / matched | 必须对齐 | 3960 / 3960 / 3960 | PASS |
| 21 点完整率 | >= 90% | 90.707% | PASS |
| PCK@20px（全部 GT） | >= 80% | 84.097% | PASS |
| PCK@20px（仅有效预测） | 记录项 | 92.713% | — |
| 2D MPJPE（仅有效预测） | <= 10 px | 8.713 px | PASS |
| 检测器 + 手选择 P95 | <= 50 ms | 23.338 ms | PASS |
| P99 / 最大值 | 记录项 | 28.094 / 39.865 ms | — |

这次 run 还绑定：

- Git full commit：`a5a53da980a60c07b5893ed42c2be34b25a63728`；
- 工作树：clean；
- 121 个源码文件的聚合 SHA-256：`ae4a6c84d0d9d82701b7f105995395d8039413db9097eed82e56d17876b87e42`；
- 该历史 run 当时的 Unity 接收脚本 SHA-256：`38f32bfeec28346ae275442a86c5971aa62d157adbec0cd276713379d9fde81f`；
- 预测、指标、配置、日志、SVG 和 checksums 均位于该 run 独立目录。

### 4.2 当前回归测试

Phase 1 clean run 仍保留 2026-08-24 不可覆盖结果；2026-08-30 Phase 1.5 当前工作树另做
完整回归：

| 环境 | 范围 | 结果 |
|---|---|---|
| `handai-intent-prediction` | 当前全部 Python 测试（baseline/Phase 1/contract/v2 shadow/网络扰动/Unity 快照） | `168 passed`，10.97 s |
| `single-right-hand-baseline` | 不含 PyTorch 可选能力的兼容回归 | `161 passed, 7 skipped`，4.56 s |
| `handai-intent-prediction` | 当前摄像头 60 帧无界面 + 后台 worker | baseline/prediction 各 60 行，输入/结果 0 丢弃；当时右手未入镜 |
| Unity 2020.3.49f1 batchmode | 编译 + loopback/invalid/乱序/过期/watchdog 动态回归 | `PHASE15_UNITY_SAFETY_BATCH_PASS`，返回码 0 |
| Unity 最小源码快照 | 7 个关键文件 manifest 哈希 + 本机外部工程对比 | 7/7 一致，无漂移 |
| 历史 Phase 1 环境 | 2026-08-24 不可覆盖 clean run | `130 passed, 7 skipped`（历史证据） |
| GitHub Actions | `windows-latest` | success |
| GitHub Actions | `ubuntu-latest` | success |

跳过项主要与环境可选能力有关；“测试通过”不应被写成真实摄像头或真机已通过。

## 5. 文件夹 Tree

以下是经过筛选的工程树。为保证可读性，不展开 H2O/FreiHAND 原始图片、Unity 第三方资源、`Library/`、`Temp/`、缓存、每次实验的完整输出和 checkpoint。

```text
D:\VR\HandAi\
├─ .github\
│  └─ workflows\
│     └─ single-hand-teleop-baseline-ci.yml   # Windows/Ubuntu CI
├─ .gitignore                                 # 数据、PDF、缓存等仓库边界
├─ LICENSE
└─ single-hand-teleop-baseline\
   ├─ AGENTS.md                               # 单右手/Unity preview 项目约束
   ├─ README.md                               # 项目总入口与运行说明
   ├─ CONTRIBUTING.md
   ├─ requirements.txt
   ├─ environment.yml                         # baseline Conda 环境
   ├─ pyproject.toml
   ├─ pytest.ini
   ├─ configs\
   │  ├─ default.yaml                         # 最保守的视觉 baseline
   │  ├─ svh_9ch_preview.yaml                 # 本地 9 通道预览，不主动发 Unity
   │  └─ unity_udp_preview.yaml               # 127.0.0.1:18080 联动配置
   ├─ schemas\
   │  └─ frame_payload.schema.json            # 逐帧 payload JSON Schema
   ├─ examples\
   │  ├─ sample_output.json
   │  ├─ sample_output_svh_9ch.json
   │  ├─ sample_prediction_diagnostics.json
   │  └─ sample_session.jsonl
   ├─ scripts\
   │  ├─ send_unity_preview_demo.py           # 无摄像头合成 UDP 验收
   │  └─ run_prediction_shadow_smoke.py       # 无摄像头真实 checkpoint smoke
   ├─ integrations\
   │  └─ unity_phase15_snapshot\              # Unity 关键源码/依赖/版本的可恢复快照
   │     ├─ snapshot_manifest.json             # 7 个文件的大小和 SHA-256
   │     ├─ Assets\Scripts\RobotControlScript.cs
   │     ├─ Assets\Editor\BaselineUdpSafetyBatch.cs
   │     ├─ Packages\manifest.json / packages-lock.json
   │     └─ ProjectSettings\ProjectVersion.txt
   ├─ src\
   │  ├─ main.py                              # 主循环和模块编排
   │  ├─ capture\
   │  │  ├─ input_source.py                   # 输入源统一接口
   │  │  ├─ webcam.py                         # OpenCV 摄像头
   │  │  └─ video_file.py                     # 固定视频回放
   │  ├─ perception\
   │  │  ├─ mediapipe_hand.py                 # 21 点及 handedness 检测
   │  │  ├─ hand_filter.py                    # 选择最高置信度 Right
   │  │  └─ landmark_quality.py               # 控制前质量门控
   │  ├─ features\
   │  │  ├─ geometry_utils.py                 # 距离、角度、归一化
   │  │  └─ hand_features.py                  # pinch/open/curl 等连续特征
   │  ├─ gesture\
   │  │  └─ rule_based_gesture.py             # open/fist/pinch/unknown + 去抖
   │  ├─ control\
   │  │  └─ control_representation.py         # 硬件无关连续控制中间层
   │  ├─ output\
   │  │  ├─ frame_payload_contract.py         # normalize + validate + timing/预测诊断 contract
   │  │  └─ json_exporter.py                  # JSON/JSONL/UDP 输出
   │  ├─ prediction\
   │  │  ├─ shadow_predictor.py               # 冻结模型、30 Hz 重采样、门控和安全回退
   │  │  └─ shadow_worker.py                  # latest-only 后台推理与独立结果队列
   │  ├─ svh\
   │  │  ├─ svh_adapter.py                    # control -> 5/9 通道 preview
   │  │  ├─ svh_command.py                    # preview 命令数据结构
   │  │  ├─ svh_layout.py                     # 9 通道顺序和参考 ticks
   │  │  ├─ svh_protocol.py                   # 真机前协议 skeleton
   │  │  ├─ svh_transport_base.py             # 传输接口
   │  │  ├─ svh_transport_mock.py             # 有界历史的 mock transport
   │  │  └─ mapping_contract.py               # 9 通道标签语义版本与哈希
   │  ├─ visualize\
   │  │  ├─ overlay_2d.py                     # 画面与状态面板拼接
   │  │  └─ status_panel.py                   # Pillow 中文状态面板
   │  └─ utils\
   │     ├─ config.py                         # YAML 和项目相对路径解析
   │     ├─ logger.py                         # 日志
   │     ├─ recent_frames.py                  # 最近帧摘要缓存
   │     └─ timer.py                          # FPS/时间工具
   ├─ experiments\
   │  ├─ freihand_eval\
   │  │  ├─ configs\freihand_eval.yaml       # 数据路径、指标和阶段门槛
   │  │  ├─ freihand\
   │  │  │  ├─ io.py                         # 标注/预测/配置 I/O
   │  │  │  ├─ projection.py                 # XYZ + K -> UV
   │  │  │  ├─ metrics.py                    # MPJPE/PCK/耗时等指标
   │  │  │  └─ report.py                     # Markdown 表格/报告
   │  │  └─ scripts\
   │  │     ├─ run_phase1_offline_acceptance.py
   │  │     ├─ run_current_pipeline_predictions.py
   │  │     ├─ evaluate_predictions.py
   │  │     ├─ inspect_annotations.py
   │  │     ├─ project_xyz_to_uv.py
   │  │     ├─ make_report_table.py
   │  │     └─ make_report_figures.py
   │  └─ intent_prediction\
   │     ├─ environment.yml                   # 独立 PyTorch/CUDA 环境
   │     ├─ configs\
   │     │  ├─ delay_injection_v1.json        # 运行前冻结的 36 场景网络扰动协议
   │     │  ├─ h2o_first_round.json
   │     │  ├─ h2o_quick_iteration.json
   │     │  └─ h2o_second_round.json
   │     ├─ intent_prediction\
   │     │  ├─ h2o_adapter.py                 # H2O pose -> 9 通道
   │     │  ├─ sequence_data.py               # NPZ、窗口和切分
   │     │  ├─ baselines.py                   # hold/linear/Kalman
   │     │  ├─ models.py                      # GRU/residual GRU/TCN/Transformer
   │     │  ├─ training.py                    # 训练、早停、预测和延迟
   │     │  ├─ metrics.py                     # 预测和运动分层指标
   │     │  ├─ gating.py                      # validation 运动门控
   │     │  ├─ experiment_runner.py           # 第一轮统一 runner
   │     │  ├─ second_round.py                # 冻结选型后加载 test
   │     │  └─ delay_injection.py             # 延迟/抖动/丢包接收端回放与 gate
   │     ├─ scripts\
   │     │  ├─ preprocess_h2o.py
   │     │  ├─ run_first_round.py
   │     │  ├─ audit_first_round_motion.py
   │     │  ├─ run_second_round.py
   │     │  └─ run_delay_injection.py
   │     └─ reports\                          # 第二轮与扰动实验正式 JSON/CSV 快照
   ├─ tests\                                  # 160 项现役环境 CLI、contract、映射、安全和预测测试
   └─ docs\
      ├─ README.md                            # 文档索引
      ├─ project_completion_manual.md         # 本说明书
      ├─ phase15_risk_hardening.md            # 二轮风险加固与动态证据
      ├─ phase1_offline_acceptance.md
      ├─ unity_handoff.md
      ├─ downstream_preview_contract.md
      ├─ intent_prediction_first_round_report.md
      ├─ intent_prediction_second_round_report.md
      ├─ intent_prediction_shadow_mode.md
      ├─ intent_prediction_delay_injection.md
      ├─ ai_scope_and_proposal_alignment.md
      └─ svh_real_hardware_calibration_table.md
```

外部但相关的本机目录：

```text
D:\VR\HandAi\datasets\H2O\
├─ downloads\                                 # 6 个所需压缩包
├─ raw\                                       # subject1...subject4 解压数据
├─ processed\cross_subject_v1\                # 历史 v1 标签，不再供现役影子模型加载
└─ processed\cross_subject_v2_open_release\   # 217 序列、113,322 帧、映射契约已冻结

D:\SVH\RoboticArm\
├─ ProjectSettings\ProjectVersion.txt          # Unity 2020.3.49f1
├─ Packages\
│  ├─ manifest.json                            # 已移除 autostreaming
│  └─ packages-lock.json
└─ Assets\
   ├─ Scenes\Hnad.unity                        # 当前联动场景
   └─ Scripts\
      ├─ RobotControlScript.cs                 # baseline UDP 接收与虚拟手应用
      ├─ RobotHand.cs / SVHFingerManager.cs    # 原工程手部/设备逻辑
      └─ RobotArm.cs / VirtualArm.cs ...       # 原工程机械臂与虚拟对象逻辑
```

## 6. 代码作用速查表

### 6.1 实时主链

| 文件/模块 | 输入 | 主要处理 | 输出 |
|---|---|---|---|
| `src/main.py` | CLI、YAML、视频帧 | 构建运行模式，串联全部模块，扩展异常安全降级 | canonical payload、GUI 和各输出通道 |
| `capture/*` | 摄像头索引或视频路径 | 统一打开、逐帧读取、释放 | `(ok, BGR frame)` |
| `mediapipe_hand.py` | BGR 图像 | RGB 转换、MediaPipe 21 点、handedness/confidence | `HandDetection[]` |
| `hand_filter.py` | 检测候选 | 过滤 `Right` 并按置信度排序 | 0 或 1 个右手检测 |
| `landmark_quality.py` | 归一化 21 点 | 掌心中心、边界与越界比例检查 | `control_ready` 判据 |
| `hand_features.py` | 2D/3D 21 点 | pinch 距离、张开比例、五指 curl | 视觉连续特征 payload |
| `rule_based_gesture.py` | 连续特征 | 可解释阈值分类和连续帧确认 | raw/stable 手势 |
| `control_representation.py` | 视觉特征、稳定手势 | 计算 grasp、pinch、support flex，抑制拳头误判 pinch | 硬件无关连续控制对象 |
| `svh_adapter.py` | control 对象、映射配置 | grasp/pinch 映射、手指耦合、通道缩放 | 5/9 通道 `svh_preview` |
| `frame_payload_contract.py` | 运行期 payload | 统一字段、值域、占位对象和严格校验 | frozen canonical payload |
| `json_exporter.py` | canonical payload | UDP 优先发送、JSON/JSONL 节流落盘、失败隔离 | 文件和本机 UDP 数据报 |
| `prediction/shadow_predictor.py` | 连续有效 `svh_9ch` preview、冻结 selection/checkpoint | 契约/SHA 校验、按时间戳重采样到 30 Hz、residual GRU、validation gate、断帧/异常回退 | 只供本机独立日志的 `prediction_diagnostics` |
| `prediction/shadow_worker.py` | 当前 canonical payload | latest-only 有界输入队列、后台推理、有界结果队列、丢弃计数和限时关闭 | 与基线帧号/时间戳对齐的独立 prediction 结果 |
| `status_panel.py` | payload | 中文字体状态面板、数组摘要和性能显示 | OpenCV 可视化面板 |

### 6.2 Preview 与协议准备层

| 文件/模块 | 作用 | 不能把它当成什么 |
|---|---|---|
| `svh_command.py` | 定义 preview 命令对象及序列化形状 | 不是真机命令执行器 |
| `svh_layout.py` | 固定 9 通道名称、顺序和 preview ticks 参考 | 不是真机已标定零位/限位 |
| `svh_protocol.py` | 保存 sync/address/endianness/checksum 等协议假设 | 不是真实设备协议验收 |
| `svh_transport_base.py` | 未来传输实现的抽象接口 | 不提供网络可靠性 |
| `svh_transport_mock.py` | 测试/预览时以有界 deque 保留最近命令 | 不发送实体 SVH，也不会随运行时长无限增长 |
| `mapping_contract.py` | 冻结 v2 标签语义版本与 SHA-256，供数据、selection、checkpoint 和运行时交叉校验 | 不等同于真实 SVH 标定 |
| `RobotControlScript.cs` | 仅监听 loopback、严格有效性/新鲜度/顺序检查、watchdog 安全张开、9 通道展开 | UDP 真机转发编译期固定关闭，不代表硬件安全链完成 |
| `BaselineUdpSafetyBatch.cs` | 在 Unity batchmode 动态复现有效包、无效包、乱序、过期和失联行为 | 是虚拟预览安全回归，不是真机验收 |

### 6.3 离线验收代码

| 文件/模块 | 作用 |
|---|---|
| `run_phase1_offline_acceptance.py` | 总编排；创建 run、执行编译/测试/预测/评估、记录环境和哈希、判断 release eligibility |
| `run_current_pipeline_predictions.py` | 把当前 MediaPipe 检测器真正跑在 FreiHAND 图片上，保存每样本关键点和耗时 |
| `freihand/io.py` | 读取配置、相机内参、XYZ、预测及标注质量诊断 |
| `freihand/projection.py` | 用相机内参把 3D GT 投影为 2D GT |
| `freihand/metrics.py` | 数量对齐、完整率、MPJPE、PCK、逐关节误差和延迟统计 |
| `freihand/report.py` + report scripts | 输出人读报告、PPT 表格和 SVG 图 |
| `send_unity_preview_demo.py` | 在没有摄像头时生成规范的连续动作 payload，可选发送本机 Unity |

### 6.4 意图预测代码

| 文件/模块 | 作用 |
|---|---|
| `preprocess_h2o.py` / `h2o_adapter.py` | 读取 H2O 右手 pose，复用主链映射生成连续 9 通道数据和 manifest |
| `sequence_data.py` | 保存/加载 NPZ，按序列构造过去 30 帧与未来多 horizon 窗口 |
| `baselines.py` | hold-last、线性外推和常速度 Kalman 对照 |
| `models.py` | absolute GRU、零初始化 residual GRU、TCN、Transformer |
| `training.py` | 固定随机性、训练、早停、checkpoint、批量预测和单窗口延迟 |
| `metrics.py` | 整体、分 horizon、分通道和已观测运动强度分层指标 |
| `experiment_runner.py` | 第一轮统一数据/训练/指标/报告入口 |
| `gating.py` | 只基于历史运动与 validation，在 hold-last 和模型之间插值 |
| `second_round.py` | validation 选型、先冻结 selection 哈希、后加载 test、自动检查 6 项门槛 |
| `delay_injection.py` / `run_delay_injection.py` | 冻结 36 个延迟/抖动/丢包场景，模拟接收端 latest-source 策略，输出 H2O/真实 JSONL 的逐场景与逐序列指标 |
| `audit_first_round_motion.py` | 把严格 Transformer 与原第一轮模型放到统一运动阈值下重审 |
| `run_prediction_shadow_smoke.py` | 无摄像头生成 36 帧连续 9 通道，并实际加载冻结 checkpoint 验证影子状态与 preview 不变 |

## 7. 怎样复现

### 7.1 baseline 环境与测试

```powershell
cd D:\VR\HandAi\single-hand-teleop-baseline
conda activate single-right-hand-baseline
python -m compileall -q src experiments\freihand_eval
python -m pytest -q
```

如果出现 `ModuleNotFoundError: No module named 'yaml'`，说明当前 Python 环境缺少 PyYAML 或用错了解释器。优先进入 `single-right-hand-baseline` 环境，再执行：

```powershell
python -m pip install -r requirements.txt
```

如果旧环境的 `pip check` 报错来自本项目未声明的 Jupyter、音频或网页包，不要把这些无关包
继续安装进 baseline。应从 `environment.yml` 新建干净环境，或在需要 PyTorch 影子能力时
使用下文的 `handai-intent-prediction` 环境。

### 7.2 无摄像头的 payload smoke

```powershell
python scripts\send_unity_preview_demo.py --max-frames 5
```

该命令只生成和校验，不发送 UDP。

### 7.3 无摄像头的 Unity 联动

先在 `Hnad.unity` 的 `RobotControlScript` Inspector 中确认：

```text
Enable Baseline Udp Preview = true
Baseline Udp Listen Port = 18080
Apply Baseline Preview To Hardware = false
Enable Legacy Gesture Snapping = false
```

点击 Play 后运行：

```powershell
python scripts\send_unity_preview_demo.py --send
```

### 7.4 Phase 1 全量验收

```powershell
python experiments\freihand_eval\scripts\run_phase1_offline_acceptance.py `
  --config experiments\freihand_eval\configs\freihand_eval.yaml `
  --split evaluation `
  --require-full-split
```

正式证据必须同时满足：`status=completed`、全部 checks 为 true、`release_eligible=true`。

### 7.5 固定视频回放

等有手机录制的右手视频后，不需要购买专用摄像头即可先做：

```powershell
python src\main.py `
  --config configs\unity_udp_preview.yaml `
  --video-file D:\path\to\right_hand_test.mp4 `
  --headless `
  --save-jsonl
```

如果同时观察 Unity，不要加 `--headless`，并确保硬件开关仍为 false。

### 7.6 H2O 预处理与模型实验

预处理在 baseline 环境：

```powershell
python experiments\intent_prediction\scripts\preprocess_h2o.py `
  --h2o-root D:\VR\HandAi\datasets\H2O\raw `
  --output-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v1
```

模型实验在独立环境：

```powershell
conda activate handai-intent-prediction

# 快速自检
python experiments\intent_prediction\scripts\run_second_round.py --synthetic-smoke

# 正式第二轮；会重新训练候选
python experiments\intent_prediction\scripts\run_second_round.py `
  --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v1
```

第一轮全模型正式复跑包含严格确定性的 TCN，曾耗时约 98 分钟；没有修改数据或配置时，不建议为了演示重复消耗时间，直接使用已冻结报告即可。

## 8. 证据文件在哪里

| 目的 | 文件 |
|---|---|
| 项目总入口 | [README.md](../README.md) |
| 本阶段总说明 | [project_completion_manual.md](project_completion_manual.md) |
| Phase 1 怎么验收 | [phase1_offline_acceptance.md](phase1_offline_acceptance.md) |
| Phase 1 最新 clean run | 本机忽略目录：`experiments/freihand_eval/outputs/phase1_acceptance/20260824T022030_928281Z_a5a53da_clean/acceptance.md` |
| Unity 重开/联动 | [unity_handoff.md](unity_handoff.md) |
| Unity 最小可恢复快照 | [snapshot README](../integrations/unity_phase15_snapshot/README.md) / [SHA-256 manifest](../integrations/unity_phase15_snapshot/snapshot_manifest.json) |
| payload 消费规则 | [downstream_preview_contract.md](downstream_preview_contract.md) |
| 第一轮研究结论 | [intent_prediction_first_round_report.md](intent_prediction_first_round_report.md) |
| 第一轮机器可读快照 | [第一轮统一运动审计 JSON](../experiments/intent_prediction/reports/first_round/20260823T063728_024071Z_motion_audit_strict_transformer.json) |
| 第二轮研究结论 | [intent_prediction_second_round_report.md](intent_prediction_second_round_report.md) |
| Phase 1.5 二轮加固 | [phase15_risk_hardening.md](phase15_risk_hardening.md) |
| 网络扰动结论 | [intent_prediction_delay_injection.md](intent_prediction_delay_injection.md)（4/6，控制参考保留 hold-last） |
| 网络扰动机器证据 | [正式 JSON](../experiments/intent_prediction/reports/delay_injection/20260829T112008_630685Z_report.json) |
| 现役 v2 机器可读快照 | [v2 正式报告 JSON](../experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_report.json)（4/6，未过 gate） |
| 历史 v1 机器可读快照 | [v1 正式报告 JSON](../experiments/intent_prediction/reports/second_round/20260823T092028_938598Z_report.json)（旧标签语义，6/6） |
| 计划书口径差异 | [ai_scope_and_proposal_alignment.md](ai_scope_and_proposal_alignment.md) |
| 真机前缺口 | [svh_real_hardware_calibration_table.md](svh_real_hardware_calibration_table.md) |
| Unity 接收脚本 | `D:\SVH\RoboticArm\Assets\Scripts\RobotControlScript.cs`；当前 SHA-256 `6a825441e9bab8d17402af2ee3f04af00cb02a98780ef81b7ae077f2526cf43f` |

## 9. 尚存风险与限制

### 9.1 真实输入域证据仍有限

目前已有一段 808 帧真实摄像头 JSONL，并已用于探索性网络扰动回放；但它不是事先冻结动作、
背景和光照的个人视频 holdout，也没有实体关节真值，所以仍不能完整回答：

- 快速挥手是否连续；
- 遮挡后能否平稳恢复；
- 室内强弱光是否导致抖动；
- 无手画面是否误检；
- 用户自己的手型、肤色、背景与 MediaPipe/H2O 的域偏移有多大。

最省成本的补法是用手机录制固定测试视频，而不是立即购买专业摄像头。

### 9.2 意图预测仍是代理任务

H2O 没有 SVH 9 通道真值。当前监督目标是把公开手姿态经规则映射后生成的代理序列，因此模型学到的是“规则映射后的短期轨迹”，不是机械手真实响应动力学。二审全量审计还确认：v2 的逐帧 raw 手势代理有 0.90% 帧会与实时去抖标签不同；pose-only 缺少内参时直接使用相机 x/y，与归一化透视标签有 95.09% 帧不同、平均每通道绝对差 0.05191。因此 v2 可复现实验，但不是与摄像头图像域完全同构的标签。

### 9.3 统计证据仍需加强

- subject4 已被第一轮看过，不是最终盲测；
- 同一序列窗口高度重叠，不能把 14,352 个窗口当作完全独立样本；
- 第二轮候选只跑一个固定 seed，前三名 validation 目标值很接近；
- 旧 v1 的 3.60%/5.73% 已因标签映射变化失去现役适用性；当前 v2 只有 2.53%/3.75%，
  未达到 3%/5% 的预注册门槛。

### 9.4 还没有证明“延迟补偿”

模型预测未来目标，不等于已经减少遥操作延迟。冻结的 36 场景回放已经完成，但 H2O
primary RMSE 只改善 1.95%、retention gate 4/6；真实 JSONL 总体改善 1.85%，条件覆盖率
65.17%，计入初始无包 tick 后端到端预测覆盖率仅 61.90%。因此当前证据明确不支持使用
“延迟补偿有效”的表述。

### 9.5 真机安全完全独立

即使 Unity 动得很好，也不能绕过以下真机条件：协议字节级确认、通道方向、零位、限位、homing、速率限制、ACK/timeout、watchdog、急停、断线安全状态和操作者风险评估。

## 10. 接下来最值得做什么

### 已完成的优先级 1：默认关闭的预测影子模式

已把选中 checkpoint 接到 Python 侧，只记录：

```text
hold-last / raw residual / gated residual
```

实现采用“当前帧 UDP 先发、非阻塞提交、后台模型计算、独立 prediction 落盘”的顺序。
`svh_preview` 未改变；历史不足、
输入 invalid、checkpoint 缺失、SHA 不符或推理异常都有确定性回退。`timing v1` 保持冻结，
可选 `prediction_diagnostics v1` 写入独立 prediction JSONL，并用同一 frame index、timestamp 与基线日志对齐。无摄像头命令、
状态表和字段边界见 [intent_prediction_shadow_mode.md](intent_prediction_shadow_mode.md)。

### 优先级 2：延迟、抖动与丢包注入（已完成）

冻结的 36 场景 H2O/JSONL 回放已经完成。H2O primary gated RMSE 改善 1.95%、q90 改善
2.67%，retention gate 4/6；真实 JSONL 总体改善 1.85%，但端到端 prediction 覆盖率仅 61.90%。
因此本项结论是继续使用 hold-last，而不是把模型接入控制。详见
[冻结回放报告](intent_prediction_delay_injection.md)。

### 优先级 3：用手机录制个人固定测试集

建议录制 open、fist、pinch、慢速连续屈伸、快速屈伸、手贴边、部分遮挡、不同背景与光照。视频和动作顺序固定，保留为后续每版回归集。这样不需要专业摄像头，也比临时对着摄像头凭感觉看更可复现。

### 优先级 4：补统计稳定性

本科项目不必同时把所有大模型做到底。优先只保留 hold-last 与 residual GRU：

- 多个 seed；
- leave-one-subject-out 或 subject bootstrap；
- 按 subject/sequence 汇总均值、方差和置信区间；
- 留一批新自录数据作为真正不参与设计的最终测试。

### 优先级 5：与导师冻结项目口径

建议书面确认：

1. MediaPipe 是否作为正式主方案，Cascade R-CNN/ResNet-101 是否只做小规模对照；
2. 最终部署目标是 VR 头显还是近端 PC；
3. 单右手是否作为可验收主线，双手是否明确降为扩展；
4. 哪批数据是最终测试，哪些指标可以写进结题材料。

当前模型只有约 17 万参数且在本机 GPU 上约 1.5 ms P95，现阶段不必把大量时间花在极致轻量化；先证明它在延迟条件和真实输入域里确实有用，更符合本科项目“尽力探索、重视学习与证据”的目标。

## 11. 可对导师使用的准确表述

可以这样说：

> 我们已经完成单右手视觉理解、连续控制表示、9 通道 Unity 虚拟手预览和 Phase 1.5 UDP 安全加固；预测支线已实现映射参数与公式闸门、30 Hz 重采样和非阻塞后台影子日志。冻结的 36 场景网络扰动回放中，H2O primary RMSE 改善 1.95%、retention gate 4/6，真实摄像头 JSONL 总体改善 1.85%且端到端预测覆盖率为 61.90%，因此控制参考继续使用 hold-last，模型不进入 Unity/UDP。

暂时不要这样说：

> 已完成双手 VR 遥操作、5G 低时延补偿和真实 SVH 控制，准确率达到 90%。

原因是“90.707%”是 FreiHAND 静态图像上 21 点预测完整率，不是项目总准确率；当前 Unity 是虚拟预览，预测结果也是 pose-only 代理任务的内部复验。

## 12. 最终验收清单

- [x] 单右手视觉 pipeline 模块化实现
- [x] 摄像头与视频文件统一输入接口
- [x] 21 点、Right 筛选、质量门控与连续特征
- [x] `control_representation` 与 `svh_preview` 9 通道
- [x] canonical payload、schema、JSON/JSONL 和 timing v1
- [x] UDP 默认关闭、本机显式开启和实时优先输出顺序
- [x] 无摄像头 Unity 合成发送器
- [x] Unity 2020.3.49f1 红错修复与 Play Mode 合成包验证
- [x] FreiHAND 3,960 张全量 clean run，`release_eligible=true`
- [x] H2O pose-only 自动预处理与跨受试者切分
- [x] 第一轮六类预测方法对照
- [x] 历史 v1 第二轮 residual GRU + validation gate，离线 6/6（仅历史语义）
- [x] 当前映射 H2O v2 重标/重训与 contract 闸门；诚实记录 gate 4/6 未通过
- [x] 历史 Phase 1 正式报告快照、PR #14 合并和双平台 CI
- [ ] 当前 Phase 1.5 二审修补 PR 与双平台 CI（以本阶段 PR 状态为准）
- [x] 默认关闭的非阻塞预测影子、30 Hz 重采样、独立日志与新 checkpoint smoke
- [x] Unity loopback/invalid/乱序/过期/watchdog 动态 batch 回归
- [x] 延迟/抖动/丢包 36 场景冻结回放；retention gate 4/6，保留 hold-last
- [x] Unity 7 个关键源码/依赖/版本文件的 Git 快照、SHA-256 和漂移回归
- [x] invalid 队列断点、mapping implementation 指纹、覆盖率分母与 H2O 标签语义审计
- [ ] 真实个人视频域验证
- [ ] 多种子/跨 subject 统计稳定性
- [ ] 真正未参与设计的最终盲测
- [ ] 实体 SVH 协议、标定与安全验收
- [ ] 双手、多手、5G、头显部署（仅在导师确认后进入范围）

---

本说明书的所有数字都应与对应 run/report 一起引用。若后续代码、配置、数据切分或 Unity 接收脚本发生变化，应新建 run 并更新本文档，而不是覆盖旧证据。
