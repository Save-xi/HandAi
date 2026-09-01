# 2026-09-01 真实摄像头与 Unity 实时验收记录

- 状态：**工程运行与人工视觉验收通过**
- 证据角色：development 实时诊断，不产生模型上线结论
- 范围：普通摄像头单右手 -> `control_representation` -> `svh_preview` -> Unity UDP 虚拟手
- 明确不包含：双手、实体 SVH、机器人本体、预测接管 UDP、真实意图或实体关节精度

## 1. 冻结身份与证据文件

本次有效运行的 `run_id`：

`20260901T020122701808Z-42096-49fd10b6`

| 证据 | 路径/身份 | SHA-256 |
|---|---|---|
| runtime manifest | `outputs/runtime_session_20260901T020122701808Z-42096-49fd10b6.json` | `6a8dc79186da9b906cfb42a0f18cffc63afa2ad83e485005c329775257527022` |
| baseline JSONL | `outputs/session_20260901T020122701808Z-42096-49fd10b6.jsonl` | `21a3af9e04ffe115be4fad78b0d84b7965e200cde3f4fe6e64ffd0b86c97292b` |
| prediction JSONL | `outputs/prediction_session_20260901T020122701808Z-42096-49fd10b6.jsonl` | `6b328e2423e0deaa1ce6bf8e72a3b64a72bb995a122f1a49d388f2a5efbb3f8c` |
| Unity timing | `unity_timing_20260901T020228312Z_e9ab3a83.json` | `c11ed21459406e70f88b3feff781b5d684a4745eee270ece0f04a95fb1901292` |
| runtime config | `configs/unity_udp_preview.yaml` | `2716014f238eac410cb7396419e4e0c8780b344a9c951665a22bd6a64205321c` |
| selection | `20260829T024400_143036Z_selection.json` | `0a46c6cd7815b3144b3d8fa86968031c130c5fd1fd96393726c9c971260a6763` |
| checkpoint | `residual_motion4.pt` | `0146d8ecdc117d9ac9d49a22f087199ca0fe1f60173fb10a2908cabf3043fee6` |
| Unity 接收脚本 | `RobotControlScript.cs` | `ce03a837854b8196359cf9ec23413b7cfa49ef812b9b72e026210c218890d4e9` |

严格评测器已经核验共享 `run_id`、manifest 状态、精确路径、字节数、行数、文件 SHA、逐帧
frame/timestamp、模型身份以及 Unity 首末帧/时间戳。未发现跨会话、重复帧、时间戳倒退、模型漂移
或宽松交集配对。

## 2. 机器可读结果

| 项目 | 结果 |
|---|---:|
| baseline / prediction 行数 | 1093 / 1093 |
| manifest 状态 | `completed` |
| 运行时长（首末 source timestamp） | 47.9393 s |
| source duration-based FPS | 22.7788 |
| source median-interval FPS | 22.6462 |
| 右手检测 | 943 / 1093（86.28%），检测结果全部为 `Right` |
| 有效控制 | 887 / 1093（81.15%） |
| stable gesture | open 348 / fist 320 / pinch 102 / unknown 323 |
| worker 结果 | 1093 提交 / 1093 完成 / 0 输入丢弃 / 0 结果丢弃 |
| prediction 状态 | predicted 755 / warming_up 132 / invalid_input 206 |
| 有效控制帧 predicted 覆盖 | 85.12% |
| 全帧 predicted 覆盖 | 69.08% |
| CUDA 推理 P50 / P95 / max | 3.01 / 6.92 / 14.87 ms |
| source.read 返回后至 UDP P50 / P95 / max | 21.82 / 25.91 / 33.84 ms |
| source.read 返回后至 Unity 应用 P50 / P95 / max | 38.36 / 53.79 / 63.91 ms |

755 条完整预测均为 3 horizon x 9 channel，全部有限；raw/gated 输出范围为 `[0, 1]`，
`raw_range_violation_count=0`。887 条有效 `svh_preview.target_positions` 也全部有限且位于 `[0, 1]`；
各屈曲通道既到达开手的 `0`，又覆盖约 `0.75–0.994` 的闭合范围。

## 3. Unity 完整计数与解释

| Unity 计数 | 值 | 解释 |
|---|---:|---|
| accepted_packet_count | 1090 | 99.73% 的 source 帧进入 Unity 主线程应用 |
| overwritten_packet_count | 3 | latest-only 接收队列在主线程应用前覆盖旧待处理包 |
| frame_gap_count | 3 | 与上述 3 个被覆盖的中间帧一致，已显式披露 |
| rejected_packet_count | 0 | 没有 payload/大小/结构非法包 |
| stale_packet_count | 0 | 没有过期或乱序包被拒绝 |
| watchdog_open_count | 1 | Python 按 `q` 退出后、Unity 停止 Play 前触发一次安全张开 |

运行内最大相邻 source timestamp 间隔为 76.1 ms，没有超过 100 ms，更没有超过 350 ms；因此
`watchdog_open_count=1` 与退出后的空窗一致，不是运行中卡顿。

Unity 使用整数毫秒记录接收时刻，本次 `udp_delivery_ms` 的 P50 为 -0.33 ms；这是发送端浮点毫秒
与接收端整数毫秒截断造成的亚毫秒量化，不表示真实负网络延迟。该局限只影响亚毫秒 UDP 诊断，
不影响包顺序、安全闸门、目标姿态或本次几十毫秒级 source-to-apply 结论。

## 4. 无效段与 warming-up 口径

本次共有 6 段 `svh_preview.valid=false`。最长一段为 frame 482–568，共 87 帧；其首末 source
timestamp 分别为 `1788228106.9245048` 和 `1788228110.1520145`，精确相差 **3.22751 s**。

这里必须使用该片段自身的 source timestamp。用全运行 22.78 FPS 简单计算 `87 / 22.78 = 3.82 s`
会把整段平均帧率误套到局部片段，也忽略“87 帧只有 86 个首末间隔”，不作为本项目计时口径。

每次无效输入都会重置预测历史；恢复后的 warming-up 实测为 24/22/22/21/22/21 个摄像头帧。
这是固定 30 Hz、966.67 ms 历史时间窗的预期行为，不是 worker 丢帧。

## 5. 用户人工视觉验收

2026-09-01，用户在本机实际 Unity Play 场景中确认：

| 人工检查 | 结果 |
|---|---|
| 张开右手时，Unity 虚拟手能够充分张开 | PASS |
| 握拳与捏合能够跟随 | PASS |
| 遮挡/失去有效输入后能够安全张开 | PASS |

这是人工视觉观察证据，与机器日志分开记录。机器 timing 的“target apply”本身不等价于已渲染画面；
本节补足了该边界，但仍不是实体机械手验收。

## 6. 当前代码的 Unity batch 复验

2026-09-01 使用 `D:\Unity\2020.3.49f1\Editor\Unity.exe` 和实际工程
`D:\SVH\RoboticArm` 重新执行：

```powershell
Unity.exe -batchmode -nographics -quit `
  -projectPath D:\SVH\RoboticArm `
  -executeMethod BaselineUdpSafetyBatch.Run `
  -logFile D:\VR\HandAi\single-hand-teleop-baseline\outputs\unity_phase15_behavior.log
```

结果：`PHASE15_UNITY_SAFETY_BATCH_PASS` 出现 1 次，无阻断编译/执行错误，进程正常退出；日志
SHA-256 为 `ef68c02f256163a9d177db0ccb3d2bfe1aac6014308460f7edb2e4cfc25cb408`。

## 7. 结论边界

本次可以签字的结论只有：

1. 单右手 baseline 到 Unity 虚拟预览的实时工程链路通过；
2. `residual_motion4` 可以在本机 RTX 4060 Laptop GPU 上异步实时运行，不阻塞 baseline；
3. 运行证据、Unity 安全计数和人工视觉结果闭环；
4. 预测继续保持 default-off shadow，不修改 `svh_preview` 或 UDP payload。

`offline_gate_passed=false` 仍然成立。该模型没有通过既定 6 项离线 retention gate 的全部项目，
所以不得宣称“延迟补偿有效”，也不得进入控制。真实摄像头域协议仍处于 development 阶段；本次结果
不能用于看完数据后反向放宽盲测门槛。实体 SVH 仍需独立完成协议、标定、限位、homing、ACK、
watchdog、急停和断线验收。
