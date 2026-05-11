# SVH 真机接入前校准表

这份文档写给准备把当前 preview 链路继续推进到真实 SVH 灵巧手的人。

先把边界说清楚：

**当前项目已经有 SVH 风格的 preview / skeleton，但还没有完成真实硬件接入。**

换句话说，现在可以做：

- 单右手视觉理解
- `control_representation`
- `svh_preview`
- Unity / mock preview 联调
- 协议形状和通道顺序的前置整理

现在还不能声称已经完成：

- 真实 SVH 协议实现
- 真实 TCP / 串口 / RS485 发送
- 实体手 homing / fault / limit / watchdog
- 真实通道标定
- 安全可用的真机控制系统

## 读这份文档前先确认

建议先完成这些前置检查：

1. 默认 baseline 能跑通。
2. `configs/svh_9ch_preview.yaml` 能输出 `svh_preview.valid=true` 的帧。
3. Unity / mock preview 能看到动作趋势。
4. 你已经理解 [下游 Preview Contract](downstream_preview_contract.md) 里的 `control_ready` / `svh_preview.valid` 门控逻辑。

如果这些还没跑通，先不要进入真机阶段。否则很容易把视觉问题、映射问题、协议问题和硬件问题混在一起。

## 当前状态总览

| 模块 | 当前状态 | 真机前结论 |
| --- | --- | --- |
| 单右手视觉检测 | 已实现 | 可以作为上游输入继续使用 |
| 连续控制表示 | 已实现 | 可以继续调参，但不绑定硬件 |
| SVH 9 通道 preview | 已实现 | 可作为真机前映射草案 |
| preview ticks | 已实现 | 只能当参考刻度，不能直接下发 |
| mock transport | 已实现 | 只记录命令，不做真实 I/O |
| TCP / 串口 / RS485 | 未实现 | 真机前必须单独实现和验证 |
| homing / fault / limit | 未实现 | 真机前必须补齐 |
| watchdog / 急停 | 未实现 | 真机前必须补齐 |

## 证据来源和可信度

当前协议和通道判断主要来自：

1. 论文或说明性材料。
2. 本地 Unity / C# 参考实现。
3. 当前 Python baseline 的 preview 代码和测试。

需要保持谨慎：

- 论文描述通常是高层结构，不等于逐字节协议规范。
- 本地 Unity / C# 代码是参考实现，不等于官方最终协议。
- Python 里的 `svh_protocol.py` 是 preview skeleton，不是真机驱动。
- 所有 ticks、checksum、长度、字节序都需要真实设备收发确认。

## 当前主假设

下面这些可以作为“联调前假设”，不能写成“已经最终确认”：

| 项目 | 当前假设 | 可信度 |
| --- | --- | --- |
| sync bytes | `0x4C 0xAA` | 中等 |
| 控制状态地址 | `0x09` | 中等 |
| 全通道目标地址 | `0x03` | 中等 |
| command payload | `40` bytes | 中等偏低 |
| command frame | `48` bytes | 中等偏低 |
| response payload | `64` bytes | 中等偏低 |
| response frame | `72` bytes | 中等偏低 |
| checksum | `CHECK1=sum`、`CHECK2=xor` preview | 低，需要实测 |
| 字节序 | 默认 little endian | 低，需要实测 |
| 链路 | `TCP/IP -> 串口服务 -> RS485 -> SVH` | 中等，需要现场确认 |
| 9 通道顺序 | 与 Unity / C# 参考实现对齐 | 中等，可作优先校准顺序 |

相关代码：

- [../src/svh/svh_protocol.py](../src/svh/svh_protocol.py)
- [../src/svh/svh_layout.py](../src/svh/svh_layout.py)
- [../src/svh/svh_adapter.py](../src/svh/svh_adapter.py)

## 9 通道顺序

当前 `svh_9ch` layout 使用这个顺序：

1. `thumb_flexion`
2. `thumb_opposition`
3. `index_finger_distal`
4. `index_finger_proximal`
5. `middle_finger_distal`
6. `middle_finger_proximal`
7. `ring_finger`
8. `pinky`
9. `finger_spread`

当前判断：

- 这个顺序已经和 Unity / C# 参考实现对齐。
- 它适合继续作为 preview 和真机前校准的首选顺序。
- 它仍然不等于真实设备通道已经实测确认。

真机前必须逐通道确认：

- 通道是否存在。
- 正方向是否正确。
- open / close 的物理方向是否一致。
- 相邻通道是否存在耦合或串扰。
- 该通道安全最小值和最大值是多少。

## preview 映射链路

当前 `svh_preview` 的生成过程是：

```text
MediaPipe 21 点
  -> hand_features
  -> control_representation
  -> svh_adapter 计算每个通道 alpha
  -> normalized target_positions
  -> target_ticks_preview
```

更短地说：

```text
视觉特征 -> 控制中间层 -> 9 通道 alpha -> 归一化位置 -> preview ticks
```

这里有两个关键点：

- `target_positions` 是 Unity / preview 更应该直接消费的值。
- `target_ticks_preview` 是最后一层参考换算，不是硬件安全命令。

## ticks 语义

当前配置里有两组 ticks：

- `svh_9ch_open_ticks`
- `svh_9ch_closed_ticks`

它们出现在：

- [../configs/svh_9ch_preview.yaml](../configs/svh_9ch_preview.yaml)
- [../configs/unity_udp_preview.yaml](../configs/unity_udp_preview.yaml)
- [../src/svh/svh_layout.py](../src/svh/svh_layout.py)

当前只能这样理解：

- 它们是 preview 阶段参考刻度。
- 它们更像来源于参考实现的 home-setting 风格区间。
- 它们没有绑定到真实设备的 homing 零位。
- 它们没有经过真实 SVH 安全限位验证。
- 它们不能直接作为真机闭合上限。

真机前必须替换或重新确认：

- 每个通道的 home / zero。
- 每个通道的 open 参考值。
- 每个通道的 closed 安全上限。
- 每个通道的正负方向。
- 每个通道的速度和步进限制。

## 真机前优先校准的配置

| 配置项 | 当前作用 | 真机前建议 |
| --- | --- | --- |
| `svh_preview_layout` | 选择 `compact5` 或 `svh_9ch` | 真机前优先围绕 `svh_9ch` 做校准 |
| `svh_9ch_open_ticks` | 9 通道 open 参考 ticks | 替换为实测 open / home 参考值 |
| `svh_9ch_closed_ticks` | 9 通道 closed 参考 ticks | 替换为实测安全闭合上限 |
| `svh_thumb_grasp_scale` | 抓握时拇指 flexion 强度 | 按真实抓握姿态调 |
| `svh_thumb_opposition_scale` | 拇指对掌程度 | 按 open / grasp / pinch 三类姿态调 |
| `svh_pinch_support_scale` | pinch 时支撑手指参与度 | 按真实 pinch 稳定性调 |
| `svh_open_spread_scale` | open 时 spread 基线 | 按张手姿态调 |
| `svh_grasp_spread_scale` | grasp 时 spread 基线 | 按抓握姿态调 |
| `svh_pinch_spread_scale` | pinch 时 spread 基线 | 按 pinch 姿态调 |
| `svh_protocol_sync_bytes` | preview sync header | 当前可先保持，真机前实测确认 |
| `svh_protocol_use_little_endian` | preview 字节序 | 必须实测确认 |
| `svh_enable_gesture_fallback` | 特征失效时是否用手势兜底 | 真机阶段建议保持 `false` |

## 协议层待确认项

| 项目 | 当前状态 | 真机前必须确认 |
| --- | --- | --- |
| `0x09 SetControlState` | 只构造 preview skeleton | 控制状态位、保留位、enable / disable 语义 |
| `0x03 SetControlCommand AllChannels` | preview 中按 9 个 `int32` 打包 | 真实通道顺序、字节序、padding 规则 |
| 地址高 4 位 | 注释里保留 channel selector 可能性 | 单通道命令时如何编码 |
| length 字段 | 跟随当前 payload / frame 假设 | length 指 payload 还是整帧 |
| `CHECK1 / CHECK2` | preview 中使用 sum / xor | 计算范围、截断规则、是否包含 header / addr / length |
| sequence / index | 尚未实现 | 是否需要递增、回环、重传 |
| response parsing | 尚未实现 | 状态、fault、位置、电流、编码器值等结构 |

建议真机前先写独立协议小工具，不要一开始就把视觉 pipeline 接进去。先能稳定发一条安全小命令，再考虑接上实时视觉。

## 链路与时序待确认项

| 层级 | 当前状态 | 真机前必须确认 |
| --- | --- | --- |
| TCP 客户端 | 未实现 | 连接、重连、超时、阻塞策略 |
| 串口服务 | 只知道链路可能存在 | IP、端口、波特率、串口参数、缓冲策略 |
| RS485 | 未接入 | 半双工收发切换、设备地址、总线冲突 |
| 初始化顺序 | 只有 preview 控制状态概念 | 上电、清 fault、enable、homing 的真实顺序 |
| 发送节奏 | 当前跟随摄像头帧率 | 真实刷新频率、节流、丢帧策略 |
| 安全策略 | 默认 invalid 不输出 preview 命令 | 仍需软硬限位、急停、速度限制 |

## 建议责任分工

| 角色 | 主要负责 |
| --- | --- |
| 视觉 / 映射侧 | `control_representation` 稳定性、手势阈值、scale 调参 |
| 协议 / 链路侧 | sync、地址、长度、字节序、checksum、TCP / 串口 / RS485 |
| 硬件 / 标定侧 | homing、零位、方向、机械限位、安全闭合范围 |
| 集成验收侧 | 单通道小步进、全通道联动、fault 恢复、invalid 帧安全性 |

这几个责任最好不要混在一个“看起来能动了”的结论里。真实硬件阶段最怕的是：视觉侧以为协议确认了，协议侧以为硬件限位确认了，硬件侧以为软件不会发危险值。

## 最小真机验收顺序

建议按下面顺序推进，每一步单独记录通过标准。

| 步骤 | 做什么 | 通过标准 |
| --- | --- | --- |
| 1 | 只连链路，不发运动命令 | TCP 稳定连接；如有 response，至少能解析头部 |
| 2 | 只测控制状态 | enable / disable 有可观察变化；不产生 fault |
| 3 | 单通道极小步进 | 指定通道朝预期方向小幅运动；无越界、无碰撞、无 fault |
| 4 | 校准 open / closed ticks | 每个通道完成 zero、方向、安全上下限确认 |
| 5 | 测全通道命令 | 多通道同时响应；无明显串扰；无 fault |
| 6 | 加入限速和平滑 | 连续命令不会抖动、猛跳或超过速度限制 |
| 7 | 接入视觉输出 | 无手 / invalid 帧不会发危险命令；open / fist / pinch 稳定 |
| 8 | 故障恢复演练 | fault、断连、急停、重连都有明确行为 |

建议每一步都保留日志：

- 发出的 raw packet
- 收到的 raw response
- 解析后的状态
- 通道目标值
- 设备实际运动观察
- 是否触发 fault

## 真机阶段必须保留的保守设计

当前 preview 里有几处设计值得保留到真机阶段：

- `control_representation`
  - 视觉和硬件命令之间必须有中间层。
- `features_valid / command_ready`
  - “能算出特征”和“可以下发命令”要分开。
- `svh_enable_gesture_fallback: false`
  - 真机阶段不要用离散手势兜底生成动作。
- invalid 帧清空目标数组
  - 无效帧应当不产生新目标。
- `target_positions` 优先于 `target_ticks_preview`
  - 先确认归一化语义和趋势，再把 ticks 绑定到真实设备。

## 当前不能写进论文或汇报里的表述

不要写：

- 已完成真实 SVH 控制。
- 已完成 SVH 通信协议。
- 当前 ticks 已可直接下发。
- 已完成实体手通道标定。
- 已完成 TCP/IP 到 RS485 的稳定控制链路。

更准确的写法：

- 已完成单右手视觉遥操作 baseline。
- 已完成硬件无关的连续控制表示。
- 已完成 SVH 风格的 9 通道 preview 映射。
- 已整理真实 SVH 接入前所需的协议、标定和安全检查项。
- 后续仍需在真实硬件上完成通信、标定、限位和安全闭环验证。

## 最后提醒

只要还没有完成真实设备上的 homing、限位、fault、watchdog、急停和 response parsing，`svh_preview` 就只能叫 preview。

这个名字保守一点，但会保护项目不在最危险的地方自信过头。
