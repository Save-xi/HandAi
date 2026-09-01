# Unity 联动交接说明

这份文档的目标很简单：

- 防止隔一段时间回来以后忘记 Unity 仿真联动怎么开
- 防止新开对话后，模型不知道当前项目已经做到哪里
- 给自己或其他协作者一个最短的重新上手入口

## 当前项目结论

当前成熟主体是：

- `single-hand-teleop-baseline`

当前已经打通的链路是：

`普通摄像头 / 视频 -> MediaPipe 右手检测 -> 连续手部特征 -> control_representation -> svh_preview -> Unity UDP 预览`

当前关于 Unity 联动的关键结论：

- 现在的 Unity 预览链已经不是“只靠 `open / fist / pinch` 三种离散手势”了。
- Python 侧会输出连续控制特征和 9 通道 `svh_preview`。
- Unity 侧优先读取 Python 发来的 `svh_preview`，再展开成虚拟手关节。
- 如果中指、无名指、小指和真人手不完全一致，优先考虑机械手自由度和关节耦合限制，而不是优先怀疑视觉检测链坏掉。

已经确认过的结论：

- “21 个视觉关键点”和“Unity 里 20 个关节”不是一一直接映射关系。
- 正确理解应该是：

`21 个关键点 -> 连续手部特征 -> 9 个执行通道 -> Unity 20 个关节展开`

## 当前本机环境

当前本机常用路径：

- Python 项目：`D:/VR/HandAi/single-hand-teleop-baseline`
- Unity 工程：`D:/SVH/RoboticArm`

当前 Unity 场景：

- `Assets/Scenes/Hnad.unity`

当前 Python 联动配置：

- [../configs/unity_udp_preview.yaml](../configs/unity_udp_preview.yaml)

## 最短重开步骤

### 1. 打开 Unity

1. 用 Unity Hub 打开 `RoboticArm`
2. 打开场景 `Assets/Scenes/Hnad.unity`
3. 在层级面板中找到挂了 `RobotControlScript` 的对象
4. 在 `Inspector` 里确认：

- `Enable Baseline Udp Preview = true`
- `Baseline Udp Listen Port = 18080`
- `Apply Baseline Preview To Hardware = false`
- `Log Baseline Preview Packets = false`
- `Enable Legacy Gesture Snapping = false`
- `Baseline Udp Watchdog Timeout Ms = 350`
- `Baseline Udp Max Packet Age Ms = 1000`
- `Allow Legacy Hardware Control = false`
- `Write Baseline Timing Summary = true`
- `Baseline Timing Sample Capacity = 4096`

说明：

- `Log Baseline Preview Packets` 默认应该关掉，不然 Unity Console 会每帧刷日志，影响观察。
- `Enable Legacy Gesture Snapping` 默认必须关掉。它是旧的离散模板吸附逻辑，会把连续动作拉回固定姿态，干扰当前预览链。
- baseline socket 只绑定 `127.0.0.1`；UDP 的真机转发在代码里固定关闭，Inspector 旧字段即使误勾也不会把包下发真机。
- `Allow Legacy Hardware Control` 是旧 COM/IP/机械臂入口总门，Phase 1/1.5 必须保持 false。
- timing 使用固定容量环形缓冲；不要为了测 P50/P95 打开逐包日志。

5. 点击 Unity 的 `Play`

### 2. 运行 Python

在 `single-hand-teleop-baseline/` 目录下运行：

```bash
python src/main.py --config configs/unity_udp_preview.yaml
```

### 3. 成功联动时应该看到什么

- OpenCV 窗口能正常预览
- 控制台可以继续打印 JSON（如果开了 `--print-json`）
- Unity 虚拟手会跟着右手动作变化
- 不只是 `open / fist / pinch`，中间连续动作也会跟着变化

需要保存实时证据时，让 Python 带 `--prediction-shadow --save-jsonl` 运行 60–90 秒。先退出 Python，
再停止 Unity Play；Unity Console 会只打印一次 `Baseline timing summary 已写入：...`。该 JSON 位于
`Application.persistentDataPath/HandAiDiagnostics/`，后续必须与同一 Python `run_id` 的 manifest 和
两份 JSONL 一起做严格配对。它测到的是 `source.read()` 返回后到 Unity 主线程应用目标，不是相机
曝光、画面渲染或真机响应时间。

## 如果联动失败，先查这些

1. Unity 场景是不是 `Assets/Scenes/Hnad.unity`
2. Unity 里 `Enable Baseline Udp Preview` 是否开启
3. Unity 监听端口是否是 `18080`
4. Python 是否真的用的是 `configs/unity_udp_preview.yaml`
5. `Enable Legacy Gesture Snapping` 有没有被误打开
6. Unity 是否卡在 Console 刷屏
7. Python 侧 `svh_preview.valid` 是否大部分时间都为 `true`

## 当前代码上已经做过的重要处理

### Python 侧

- [../src/control/control_representation.py](../src/control/control_representation.py)
  - 连续控制不再被离散手势过早截断
  - 握拳时对误触发 pinch 做了抑制
  - Unity/SVH 预览配置会在稳定 `open` 中按 `hand_open_ratio` 连续释放残余 curl；原始 `finger_curl` 不改写

- [../src/svh/svh_adapter.py](../src/svh/svh_adapter.py)
  - support fingers 的权重做了增强
  - `svh_9ch` 映射比早期版本更连续

- [../src/visualize/status_panel.py](../src/visualize/status_panel.py)
  - OpenCV 状态面板中文显示改成了 `Pillow + 系统字体`

- [../src/output/json_exporter.py](../src/output/json_exporter.py)
  - 已支持 Unity 本地 UDP 发送

### Unity 侧

- `RobotControlScript` 现在优先吃 Python 侧的 `svh_preview`
- `Log Baseline Preview Packets` 默认关闭
- `Enable Legacy Gesture Snapping` 默认关闭
- 对 middle / ring / little 做过轻微增益补偿

### 2026-08-29 Phase 1.5 UDP 安全加固

- 监听地址从所有网卡收敛为 `127.0.0.1`，并拒绝空包和超过 32 KiB 的包。
- canonical preview 必须同时满足 `detected/control_ready/preview.valid/control.valid/features_valid/command_ready`。
- payload 已带 `svh_preview` 但为 invalid/no-hand 时立即张开，不允许再绕入旧 fallback。
- frame index/timestamp 倒序、重放、过期和明显未来包不会改写当前目标。
- 350 ms 没有可接受的新包时，watchdog 把虚拟手回到全张开姿态。
- baseline UDP 编译期固定只驱动虚拟手；旧 COM/IP 入口由 `allowLegacyHardwareControl=false` 隔离。
- Unity 2020.3.49f1 batchmode 已通过真实 loopback socket 行为回归，标记为
  `PHASE15_UNITY_SAFETY_BATCH_PASS`。测试脚本在
  `D:\SVH\RoboticArm\Assets\Editor\BaselineUdpSafetyBatch.cs`。

### 2026-09-01 运行证据加固

- Python baseline/prediction JSONL 共享一个文件名级 `run_id`，另有原子
  `runtime_session_<run_id>.json` 冻结路径、SHA、行数、模型身份与 worker 计数。
- Unity 对 Python post-capture、UDP delivery、主线程排队和 source-to-target apply 使用每项 4096
  样本的有界环形缓冲，退出 Play 时只写一次 P50/P95/max 摘要。
- `run_id` 不进入 UDP payload；预测仍是 default-off shadow，不改变 `svh_preview`。

完整证据和仍未完成项见
[Phase 1.5 二轮风险加固完成记录](phase15_risk_hardening.md)。
2026-09-01 已用真实摄像头和 CUDA 影子完成 1,093 帧严格配对，且用户确认充分张开、
握拳/捏合跟随、遮挡后安全张开三项均正常；详见
[真实摄像头与 Unity 实时验收记录](live_runtime_acceptance_20260901.md)。

由于完整 Unity 工程不是 Git 仓库，当前已把两份关键 C#、对应 `.meta`、包清单和
`ProjectVersion.txt` 固定到
[Unity Phase 1.5 最小源码快照](../integrations/unity_phase15_snapshot/README.md)。
快照 manifest 会校验 7 个文件的 SHA-256；本机 `D:\SVH\RoboticArm` 存在时，测试还会
检查运行工程是否已经和仓库快照漂移。正常联动仍从原 Unity 工程启动，不从快照目录启动。

### 2026-08-28 实拍张手修正

外接摄像头实拍日志曾出现“握紧正常、张开后虚拟手仍半握”的现象。排查结果是：

- Unity 端代码会把较小目标线性映射为较小关节角，未发现阻止回退或强制保持握拳的逻辑。
- 单目关键点在张手时仍给出了偏大的拇指/小指 `finger_curl`，导致旧映射的
  `grasp_close` 和前 8 个屈曲通道没有回到接近零。
- `configs/unity_udp_preview.yaml` 现已显式启用 `control_open_release_enabled`。
  稳定手势为 `open` 时，`hand_open_ratio` 从 `0.85` 到 `0.95` 会让控制用
  `finger_flex` 连续衰减到零；`pinch`、`fist` 和原始 `finger_curl` 不受改写。

如果以后换摄像头、拍摄距离或手型后需要调节，只调整：

- `control_open_release_start_ratio`：从何时开始释放
- `control_open_release_full_ratio`：何时完全张开

不要先打开 `Enable Legacy Gesture Snapping`，它会重新引入离散姿态吸附。

## 现在最需要记住的非代码结论

这条是最重要的：

- 如果现在看到的是“中指、无名指、小指不能像真人一样完全独立”，这很大概率是机械手自由度设计问题，不一定是视觉检测链或 JSON 映射链的问题。

也就是说，后面如果还要继续调，目标应该更像：

- 提高观感一致性
- 降低明显违和
- 调整 9 通道到 20 关节的视觉展开效果

而不是执着于“让机械手完全像真人手逐指独立”。

## 新开对话时的开场提示词

如果你以后新开一个对话，最省事的开场可以直接贴这个：

```text
项目在 D:\VR\HandAi\single-hand-teleop-baseline。
Unity 工程在 D:\SVH\RoboticArm。
请先阅读 single-hand-teleop-baseline/README.md 和 single-hand-teleop-baseline/docs/unity_handoff.md。
当前重点是单右手 baseline 和 Unity UDP 仿真联动，不是双手，也不是真机控制。
我需要你先根据文档恢复上下文，再继续当前任务。
```

如果你想让对方更快进入“代码回看”模式，可以用这个版本：

```text
项目在 D:\VR\HandAi\single-hand-teleop-baseline，Unity 工程在 D:\SVH\RoboticArm。
请先看：
1. single-hand-teleop-baseline/README.md
2. single-hand-teleop-baseline/docs/unity_handoff.md
3. single-hand-teleop-baseline/configs/unity_udp_preview.yaml

当前已经打通：摄像头右手检测 -> control_representation -> svh_preview -> Unity UDP 预览。
请先回看代码和文档，再继续，不要先假设是双手或真机链路。
```

## 下次回来时建议先看的文件

优先级从高到低：

1. [../README.md](../README.md)
2. [unity_handoff.md](unity_handoff.md)
3. [../configs/unity_udp_preview.yaml](../configs/unity_udp_preview.yaml)
4. [../src/control/control_representation.py](../src/control/control_representation.py)
5. [../src/svh/svh_adapter.py](../src/svh/svh_adapter.py)
6. [Unity Phase 1.5 最小源码快照](../integrations/unity_phase15_snapshot/README.md)

## 一句话总结

如果你下次又忘了，就记住：

先开 `Hnad.unity`，确认 `UDP 预览开 / legacy snapping 关 / 端口 18080`，再跑：

```bash
python src/main.py --config configs/unity_udp_preview.yaml
```
