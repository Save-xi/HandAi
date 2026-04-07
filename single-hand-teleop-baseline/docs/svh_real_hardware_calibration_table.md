# SVH 真机接入前的校准对照表

## 1. 文档目的

这份文档只服务于一个目标：

- 把当前 `single-hand-teleop-baseline` 里的 `svh preview / skeleton` 与未来真实 SVH 接入之间仍然缺少的校准项拆清楚

它不是：

- 真机联调说明书
- 最终官方协议文档
- 对官方 SDK 的替代品

当前 baseline 仍然只覆盖：

- 单右手视觉理解
- `control_representation`
- `svh_preview / skeleton`
- `mock transport`

与真实硬件相关的 TCP/IP、串口服务、RS485、实体手 homing、限位、故障恢复和状态反馈，都还没有完成接入。

## 2. 证据来源与可追溯性

当前判断主要来自两类材料：

1. 论文和说明性文字
2. 本地 Unity / C# 参考实现

需要特别强调：

- 论文文字更偏高层描述，不等于逐字节协议规范
- 本地 Unity / C# 代码是项目侧参考实现，不等于最终官方权威实现
- 因此这里很多结论都只能写成“当前假设”或“联调前默认值”，不能写成“已经最终确认”

## 3. 当前已经基本对齐的主假设

下面这些内容目前可以作为“方向基本找对了”的工程假设，但仍不应当被写成最终协议事实：

| 项目 | 当前 baseline | 当前更准确的表述 |
| --- | --- | --- |
| sync bytes | `0x4C 0xAA` | 可作为当前 preview 阶段主假设 |
| 控制状态地址 | `0x09` | 可作为当前 preview 阶段主假设 |
| 全通道目标位置地址 | `0x03` | 可作为当前 preview 阶段主假设 |
| checksum 形式 | `CHECK1=sum`、`CHECK2=xor` preview | 仅可作为联调前参考默认值 |
| 链路思路 | `TCP/IP -> 串口服务 -> RS485 -> SVH` | 仅可作为联调前链路假设 |
| 9 通道名称和顺序 | 已与 Unity / C# 参考实现对齐 | 可作为 preview 和真机前校准的首选顺序，但不能替代实测 |

## 4. 关于 `40 / 64` 长度的当前理解

当前 baseline 在 [svh_protocol.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/svh/svh_protocol.py) 里采用的是“基于本地 C# 参考代码推断出的 framing 假设”：

- command payload: `40`
- command frame: `48`
- response payload: `64`
- response frame: `72`

更准确的表述应该是：

- 这是基于当前参考实现推断出来的 framing 假设
- 与论文里的“包长度”描述之间仍可能存在解释差异
- 真机接入前，仍需要通过真实收发进一步确认

## 5. 9 通道顺序假设

当前仓库在 `svh_9ch` 模式下使用的顺序是：

1. `thumb_flexion`
2. `thumb_opposition`
3. `index_finger_distal`
4. `index_finger_proximal`
5. `middle_finger_distal`
6. `middle_finger_proximal`
7. `ring_finger`
8. `pinky`
9. `finger_spread`

更准确的理解是：

- 当前 9 通道顺序已经与参考 Unity / C# 实现对齐
- 这很适合作为 preview 层和真机前校准的首选顺序
- 但它仍不能替代设备实测确认

同时要注意：

- 这些名称是当前软件层的 preview 命名
- 它们不等价于已经证明每个通道都和真实机械关节一一对应

## 6. ticks 语义声明

当前 [svh_layout.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/svh/svh_layout.py) 与 [svh_9ch_preview.yaml](/D:/VR/HandAi/single-hand-teleop-baseline/configs/svh_9ch_preview.yaml) 中的：

- `svh_9ch_open_ticks`
- `svh_9ch_closed_ticks`

都应该被理解为：

- 当前 preview 阶段的参考刻度
- 来源于参考实现的 home-setting 风格区间
- 不是已经绑定真实 homing 零位的最终编码器命令值
- 也不是已经过真实 SVH 安全验证的最终上下限

## 7. 当前 preview 映射的优先级

当前 `svh_9ch` 模式里的命令合成顺序是：

1. 先从 `control_representation` 或 `gesture_fallback` 计算每个通道的 `alpha`
2. 再把 `alpha` 映射成归一化 `target_positions`
3. 最后再映射成 `target_ticks_preview`

也就是：

```text
alpha -> normalized position -> preview ticks
```

所以更准确的说法是：

- scale 先作用在 `alpha`
- `open_ticks / closed_ticks` 在最后一层负责把 `alpha` 映射到 ticks

## 8. 真机前最需要继续校准的参数

如果未来继续沿用这个 baseline，下面这些配置项最值得优先按设备重新校准：

| 配置项 | 当前作用 | 真机前建议处理方式 |
| --- | --- | --- |
| `svh_preview_layout` | 选择 `compact5` 或 `svh_9ch` | 真机侧建议优先围绕 `svh_9ch` 校准 |
| `svh_9ch_open_ticks` | 9 通道 open 参考 ticks | 替换为真实设备的 home/open 参考值 |
| `svh_9ch_closed_ticks` | 9 通道 closed 参考 ticks | 替换为真实设备的安全闭合上限 |
| `svh_thumb_grasp_scale` | 调整 grasp 时拇指 flexion 强度 | 按真实抓握姿态重新校准 |
| `svh_thumb_opposition_scale` | 调整拇指对掌程度 | 按 open / grasp / pinch 三类姿态重新校准 |
| `svh_pinch_support_scale` | 调整 pinch 时支撑手指参与度 | 按真实 pinch 稳定性重新校准 |
| `svh_open_spread_scale` | open 时的 spread 基线 | 按张手姿态重新校准 |
| `svh_grasp_spread_scale` | grasp 时的 spread 基线 | 按抓握姿态重新校准 |
| `svh_pinch_spread_scale` | pinch 时的 spread 基线 | 按 pinch 姿态重新校准 |
| `svh_protocol_sync_bytes` | preview sync header | 当前可先保持不变，真机前再确认 |
| `svh_protocol_use_little_endian` | preview 字节序 | 必须通过真实收发确认 |
| `svh_enable_gesture_fallback` | continuous features 失效时是否允许 fallback | 真机阶段建议继续保持 `false` |

## 9. Packet / 协议层仍需确认的部分

当前 [svh_protocol.py](/D:/VR\\HandAi\\single-hand-teleop-baseline/src/svh/svh_protocol.py) 仍然只是更贴近论文和 Unity / C# 驱动的 preview skeleton。下面这些都还需要真机前继续确认：

| 项目 | 当前状态 | 真机前必须确认 |
| --- | --- | --- |
| `0x09 SetControlState` payload | 只把首字节写入 40-byte payload preview | 控制状态位含义、保留位、是否还有额外字段 |
| `0x03 SetControlCommand AllChannels` payload | 预览时按 9 个 `int32` 打包 | 真实字节序、真实通道顺序、padding 规则 |
| 地址高 4 位语义 | 只在注释里保留“可能用于 channel selector” | 单通道命令时高 4 位到底如何编码 |
| `length` 字段 | 当前跟随 payload / frame 假设 | 论文里的“包长度”到底指 payload 还是整帧 |
| `CHECK1 / CHECK2` | 已做 `sum/xor` preview | 计算范围、截断规则、是否包含地址和长度等 |
| `index` / sequence | 尚未做真实收发序号语义 | 是否要求递增、回环、重传相关规则 |
| response parsing | 还没有解析真实返回帧 | 状态、fault、位置、电流、编码器值等真实结构 |

## 10. 链路与时序仍需确认的部分

就算 packet 字节已经对了，真实链路仍然需要单独校准：

| 层级 | 当前状态 | 真机前必须确认 |
| --- | --- | --- |
| TCP 客户端 | 尚未实现 | 连接、重连、超时、阻塞策略 |
| 串口服务 | 只知道链路存在 | IP、端口、波特率、串口参数、缓冲策略 |
| RS485 | 尚未接入 | 半双工时序、收发切换、设备地址冲突 |
| 初始化顺序 | 只有 preview `SetControlState` 概念 | 上电、清 fault、enable channels、homing 的真实顺序 |
| 发送节奏 | 当前只受摄像头帧率驱动 | 真实刷新频率、去抖、节流、丢包策略 |
| 安全策略 | 当前更保守，默认不发 fallback | 真机仍需速度/步进限制、急停、越界保护 |

## 11. 建议的责任分工

为了避免之后出现“都以为别人确认过了”的情况，建议至少按下面几类角色拆责任：

| 角色 | 主要负责确认内容 |
| --- | --- |
| 视觉 / 映射侧 | `control_representation` 是否稳定、`open/fist/pinch` 映射是否符合操作意图、各类 scale 如何调 |
| 协议 / 链路侧 | sync bytes、地址、长度、endianness、checksum、TCP / 串口服务 / RS485 收发 |
| 硬件 / 标定侧 | homing、零位、正负方向、机械限位、安全闭合范围、各通道 open/closed ticks |
| 集成验收侧 | 单通道小步进、全通道联动、fault 恢复、无效帧下的安全性 |

## 12. 真机接入前的最小验收清单

建议至少按下面顺序做验收，并明确“什么算通过”：

| 步骤 | 验收内容 | 通过判据 |
| --- | --- | --- |
| 1 | 只连链路，不发运动命令 | TCP 稳定连接；如有 response，至少能收到可解析头部 |
| 2 | 只测 `SetControlState` | 控制器状态有可观察变化；没有新增 fault |
| 3 | 只测单通道小步进 | 指定通道朝预期方向发生小位移；无越界、无碰撞、无 fault |
| 4 | 校准 `open_ticks / closed_ticks` | 每个通道完成 zero / home / direction / 安全上下限确认 |
| 5 | 再测 `AllChannels` | 多通道同时响应；没有明显串扰；无 fault；姿态组合符合预期 |
| 6 | 最后接入视觉输出 | 无手 / invalid 帧不发危险命令；`open/fist/pinch` 稳定；过渡帧不产生明显错误动作 |

## 13. 当前值得保留的 preview 设计

在进入真机前，下面这些设计值得继续保留：

- `control_representation`
  - 把视觉特征和硬件命令隔开
- `features_valid / command_ready`
  - 把“特征可算”和“命令可发”分层
- `svh_enable_gesture_fallback: false`
  - 更接近硬件阶段需要的保守策略
- `svh_9ch` + `target_ticks_preview`
  - 足够接近 SVH，但又没有假装已经完成真机接入

## 14. 当前仍不能声称已经完成的部分

基于当前仓库状态，仍然不能声称：

- 已完成真实 SVH 协议实现
- 已完成真实 TCP/IP 到串口服务的发送
- 已完成真实 RS485 联调
- 已完成实体手通道标定
- 当前 preview ticks 可以直接作为真机命令

更准确的说法应该是：

- 现在已经有一套更接近 SVH 的 `preview / skeleton`
- 同时也已经把未来真实接入前仍需继续校准的关键点、公式和验收项拆清楚了
