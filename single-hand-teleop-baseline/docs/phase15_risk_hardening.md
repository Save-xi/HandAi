# Phase 1.5 二轮风险加固完成记录

日期：2026-08-29
Python 工程：`D:\VR\HandAi\single-hand-teleop-baseline`
Unity 工程：`D:\SVH\RoboticArm`

## 1. 范围与结论

本轮严格保持当前主线：

```text
普通摄像头 / 右手视频
  -> 单右手检测
  -> control_representation
  -> svh_preview
  -> 本机 Unity UDP 虚拟预览
```

没有扩成双手，没有接实体 SVH、AUBO、串口、RS485 或 5G。旧真机代码只做默认隔离，
没有把它包装成“已经可用”。

本轮完成了 DeepSeek 二轮审计指出的主要工程风险清理：Unity UDP 接收安全、失联安全姿态、
乱序/过期包、invalid 兜底绕过、mock 内存增长、runtime/schema 契约偏差、运行产物污染、
影子推理阻塞、摄像头帧率与训练时间尺度不一致，以及模型/映射版本错配。

当前总判断：

- 单右手 Unity 虚拟预览链经过了更严格的安全加固和自动回归。
- prediction shadow 已变成非阻塞后台观察层，仍默认关闭且永不进入 UDP。
- 旧模型因映射不一致被主动淘汰；已用当前 open-release 映射重新生成 H2O v2 标签并重训。
- v2 模型有改善，但没有通过预注册的全部离线门槛，因此只能作为影子诊断，不得宣称
  “延迟补偿已有效”。

## 2. Unity 侧完成项

核心文件：`D:\SVH\RoboticArm\Assets\Scripts\RobotControlScript.cs`

### 2.1 本机限定与输入边界

- UDP socket 只绑定 `127.0.0.1`，不再监听所有网卡。
- 单包大小限制为 32 KiB，空包和超限包直接拒绝。
- baseline UDP 编译期硬门固定为 `false`，即使 Inspector 旧字段被勾选，UDP 也只驱动虚拟手。

### 2.2 严格有效性门

canonical `svh_preview` 只有同时满足以下条件才应用：

```text
detected
control_ready
svh_preview.enabled
svh_preview.valid
control_representation.valid
control_representation.features_valid
control_representation.command_ready
9 个有限 target_positions
```

只要 payload 已显式带有 `svh_preview`，但它是 invalid/no-hand，就立即回到 9 通道全零的
虚拟手张开姿态，不再落入旧连续特征 fallback。旧 fallback 只兼容真正缺少
`svh_preview` 字段的老 payload，而且同样要求完整控制门。

### 2.3 包时序与失联

- 拒绝负帧号、非法时间戳、过期包、未来时钟偏差过大的包。
- 拒绝 frame index 或 source timestamp 不递增的乱序/重放包。
- Python 重启回到 frame 0 时，仅在 source timestamp 更新后允许重置。
- 连续 350 ms 没有可接受的新包时，watchdog 只把 Unity 虚拟手切到全张开安全姿态。
- 覆盖包、frame gap、拒绝包、过期/乱序包、watchdog 张开次数都有独立计数。

### 2.4 旧真机入口隔离

新增 `allowLegacyHardwareControl=false` 总门。默认状态下：

- `Init(COM, IP, ...)` 会忽略 COM/IP，不创建旧 SVH/串口/机械臂网络对象；
- `ApplyRobotHandTargets(true)` 会降级为只更新 Unity 虚拟手；
- 本轮没有测试或启用实体硬件。

### 2.5 Unity 自动行为验收

新增：`D:\SVH\RoboticArm\Assets\Editor\BaselineUdpSafetyBatch.cs`

Unity 2020.3.49f1 batchmode 已动态验证：

1. 启动默认张开；
2. 有效包产生非零虚拟目标；
3. invalid/no-hand 立即张开；
4. 乱序和过期包不能改写目标；
5. watchdog 超时张开；
6. socket 实际绑定 IPv4 loopback；
7. 通过真实本机 UDP socket 发送有效/无效 JSON 后，姿态按规则切换。

证据日志：`outputs/unity_phase15_behavior.log`，包含
`PHASE15_UNITY_SAFETY_BATCH_PASS`，Unity 返回码为 0。

## 3. Python 运行时完成项

### 3.1 mock 内存有界

`MockSvhTransport.sent_commands` 从无限 list 改成有界 deque，默认只保留最近 32 条；
`recorded_count` 仍独立统计整个会话累计发送数。

实测发送 50,000 次：

| 指标 | 旧实现 | 当前实现 |
|---|---:|---:|
| 保留命令数 | 50,000 | 32 |
| tracemalloc 峰值 | 约 76.25 MiB | 约 0.029 MiB |

### 3.2 runtime contract 与 JSON Schema 收紧

运行时现在与 `additionalProperties: false` 对齐，拒绝以下未知字段：

- payload 顶层；
- `finger_curl`；
- `control_representation` 及其 `finger_flex`；
- `svh_preview`；
- `protocol_hint`；
- `timing`；
- `prediction_diagnostics`。

normalize 不再静默吞掉未知键，而是保留给 validator 明确报错。现有 3 个真实 JSONL、
共 1,920 行同时通过 runtime validator 和 Draft 2020-12 JSON Schema，拒绝数均为 0。

### 3.3 配置启动前硬校验

`load_config` 会在打开摄像头或创建输出前检查：

- Unity UDP 只能使用 loopback；
- 端口、尺寸、队列、flush、帧率等必须为合法正值；
- `svh_preview_layout` 与通道数必须是 `compact5/5` 或 `svh_9ch/9`；
- open-release start 必须小于 full；
- 当前 `svh_transport` 只能是 mock；
- 9 通道 ticks 必须恰好为 9 个整数；
- 运行期 last-frame JSON 不允许覆盖受版本控制的 `examples/`。

### 3.4 输出文件风险

- 运行期 last JSON 改写到 `outputs/latest_frame.json` 或 `outputs/latest_svh_9ch.json`；
  `examples/` 只保留冻结示例。
- last JSON 使用同目录临时文件 + 原子 replace，读取方不会看到半截 JSON。
- JSONL 文件名加入微秒、PID 和随机 nonce，同秒/并行启动不会误追加到同一文件。

## 4. prediction shadow 完成项

### 4.1 非阻塞后台 worker

当前顺序是：

```text
当前 canonical baseline payload
  -> 立即发送原始 payload 给 Unity UDP
  -> 非阻塞提交到 latest-only 后台队列（容量 1）
  -> baseline last JSON / session JSONL 正常写出

后台线程
  -> 历史更新、模型推理和 frozen gate
  -> 独立写入 latest_prediction_shadow.json
  -> 可选写入 prediction_session_*.jsonl
```

模型结果不再附回当前 baseline 日志，也不可能进入 UDP。baseline 和 prediction 日志都用
`frame_index`/`source_frame_index` 对齐。输入队列和结果队列都有上限；模型落后时丢等待中的旧帧，
保留最新帧，不累计实时延迟。

注入 50 ms 慢推理的单元测试证明 `submit` 小于 20 ms，真正推理在后台完成。

### 4.2 30 Hz 时间戳重采样

模型仍使用 30 帧历史，但不再把“不规则的 30 个摄像头帧”直接当作 1 秒。原始有效帧先保存在
有界时间窗中，再按 timestamp 线性插值到固定 30 Hz、966.67 ms 的 30 帧输入。

对 2026-08-28 的真实摄像头日志回放：

- 808 帧，712 帧 `svh_preview.valid=true`；
- 458 帧产生完整预测，254 帧 warming up，96 帧 invalid；
- 原始帧率 p50 约 12.49 Hz；
- 所有 predicted 输入的重采样跨度均为 966.67 ms；
- CUDA 推理 p50/p95/max 约为 1.01/1.34/1.62 ms。

### 4.3 映射契约与旧模型淘汰

新增 `svh9-label-v2-open-release` 语义契约。指纹只覆盖会改变 9 通道标签的手势阈值、
control refs、open-release、SVH layout/scales 和固定通道顺序，不受日志路径、UDP 端口等无关项影响。

H2O manifest、selection、checkpoint 和实时配置必须具有相同：

- `mapping_contract_version`；
- `mapping_contract_sha256`；
- checkpoint `data_contract`；
- selection/checkpoint SHA；
- report 引用的 selection SHA。

旧 v1 selection 没有 `data_contract`，并且旧映射指纹与当前 open-release 不同，因此会安全返回
`initialization_error`，不会悄悄加载。

## 5. H2O v2 重标与重训结果

新数据目录（不覆盖 v1）：

`D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release`

数据保持相同跨受试者划分：

| split | sequences | frames |
|---|---:|---:|
| train | 105 | 54,243 |
| validation | 60 | 28,631 |
| test | 52 | 30,448 |

映射契约 SHA：

`453c18c3ae851c325cd3b882409800c074f97f7f7ad898865d1d4895b2e318a0`

第二轮仍遵守“validation 选型并写盘冻结 selection，之后才加载 test”。选中
`residual_motion4`，checkpoint SHA：

`0146d8ecdc117d9ac9d49a22f087199ca0fe1f60173fb10a2908cabf3043fee6`

test 对比：

| 指标 | hold-last | v2 gated | 相对变化 |
|---|---:|---:|---:|
| MAE | 0.02193114 | 0.02144826 | 改善 2.20% |
| RMSE | 0.06948886 | 0.06773110 | 改善 2.53% |
| P95 | 0.08279563 | 0.07816664 | 改善 5.59% |
| 越界率 | 0 | 0 | 持平 |

高运动 q90 RMSE 改善 3.75%，单窗口推理 p95 约 0.95 ms。

预注册要求是总体 RMSE 至少改善 3%、q90 RMSE 至少改善 5%。这两项没有过线，其他四项通过，
因此 `offline_gate_passed=false`。不得为了让结果“好看”而在看到 test 后放宽门槛或重选方案。

机器可读证据：

- `experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_report.json`
- `experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_selection.json`
- `experiments/intent_prediction/reports/second_round/20260829T024400_143036Z_validation_candidates.csv`

## 6. 本轮验证汇总

| 检查 | 结果 |
|---|---|
| `handai-intent-prediction` 全量 pytest | `160 passed` |
| baseline 环境全量 pytest | `153 passed, 7 skipped` |
| Python compileall | PASS |
| `handai-intent-prediction` 环境 pip check | `No broken requirements found` |
| Unity 2020.3.49f1 脚本编译 | 返回码 0 |
| Unity 动态安全/loopback UDP 回归 | PASS 标记，返回码 0 |
| Unity 最小源码快照 | 7/7 文件哈希通过，且本机外部工程无漂移 |
| 当前摄像头无界面 60 帧短跑 | 60/60 baseline + 60/60 prediction 记录，worker 0 丢弃 |
| 当前短跑手部状态 | 60 帧均 invalid（运行时没有把右手放入画面，不是算法验收） |
| 旧真实日志 v2 重放 | 458 predicted，固定 30 Hz 时间窗 |
| 50,000 次 mock 压测 | 只保留 32 条，峰值约 0.029 MiB |
| 1,920 行历史 JSONL runtime/schema 双校验 | 0 拒绝 / 0 拒绝 |

## 7. 仍未完成与建议

1. 后续延迟、抖动、丢包冻结回放已完成，但 retention gate 仍为 4/6；所以仍不能说预测改善了遥操作延迟。
2. 当前 v2 模型没有通过全部预注册门槛；保留作本科阶段学习和影子分析，不进入控制。
3. 当前 60 帧摄像头短跑没有右手入镜，不能替代用户手动 open/fist/pinch/连续动作验收。
4. Unity batchmode 证明脚本行为与本机 UDP 回环，但不替代用户在正式场景中的视觉观感验收。
5. Unity 完整工程当前不是 Git 仓库；已把接收脚本、batch 验收脚本、包清单和编辑器版本
   作为 [最小可恢复快照](../integrations/unity_phase15_snapshot/README.md) 纳入本仓库，
   场景和第三方资源仍由原 Unity 工程保存。
6. 实体 SVH 安全仍是独立阶段：协议、标定、限位、homing、ACK、watchdog、急停均未验收。

后续冻结实验结果见 [延迟、抖动、丢包回放报告](intent_prediction_delay_injection.md)：H2O
primary RMSE 改善 1.95%，真实 JSONL 改善 1.85%，均不足以改变控制参考；应如实保留
hold-last，不强行上线预测。
