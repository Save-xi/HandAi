# SVH 真机接入前校准对照表

## 1. 目的

这份文档只服务于一个目标:

- 把当前 `single-hand-teleop-baseline` 里的 `svh preview / skeleton` 和未来真实 SVH 接入之间还差什么讲清楚

它不是:

- 真机联调说明书
- 最终官方协议文档
- 对官方 SDK 的替代

当前 baseline 仍然只做到:

- 单右手视觉理解
- `control_representation`
- `svh` preview / skeleton
- `mock transport`

真实硬件相关的 TCP/IP, 串口服务器, RS485, 实体手 homing, 限位, 故障恢复, 状态反馈都还没有接。

## 2. 证据来源与可追溯性边界

这份表里的判断来自两类材料:

1. 论文文字描述
   - `D:/VR/181910731_陈致一_基于视觉的五指灵巧手控制技术研究.docx`
2. 本地 Unity/C# 参考实现
   - `D:/SVH/RoboticArm/Assets/Scripts/SVH/SVHConstants.cs`
   - `D:/SVH/RoboticArm/Assets/Scripts/SVH/SVHSerialInterface.cs`
   - `D:/SVH/RoboticArm/Assets/Scripts/SVH/SVHController.cs`
   - `D:/SVH/RoboticArm/Assets/Scripts/SVHFingerManager.cs`

这里要特别强调:

- 论文文字是高层描述，不等于逐字节协议规范
- 本地 Unity/C# 代码是项目侧参考实现，不等于官方最终权威实现
- 所以这份表里很多内容只能写成:
  - 当前主假设
  - 联调前参考默认值
  - 真机前必须继续确认

不应该写成“已经定论”。

### 2.1 参考实现快照冻结说明

当前这份表不是基于“永远不变的官方源”，而是基于一次本地参考实现快照整理出来的。

本次整理时记录到的本地文件修改时间是:

- `SVHConstants.cs`: `2022-05-10 11:24:55`
- `SVHSerialInterface.cs`: `2022-04-01 15:29:33`
- `SVHController.cs`: `2022-04-01 10:56:44`
- `SVHFingerManager.cs`: `2022-05-11 15:02:38`

当前 HandAi 仓库头部提交是:

- `36be319`

因此这里更准确的理解是:

- 当前结论基于 `2026-04-01` 这次读取到的本地参考实现快照
- 如果 Unity/C# 参考代码后续发生改动，应重新核对本表
- 如果未来拿到更权威的官方协议资料，本表应优先服从更高可信度证据

## 3. 已经对齐到参考实现的主假设

下表里的内容可以理解为“当前方向基本找对了”，但都还没有资格写成最终协议事实。

| 项目 | 当前 baseline | 参考来源 | 当前建议表述 |
| --- | --- | --- | --- |
| sync bytes | `0x4C 0xAA` | 论文 + `SVHConstants.cs` | 可作为当前主假设 |
| 控制状态地址 | `0x09` | 论文 + `SVHConstants.cs` | 可作为当前主假设 |
| 全通道目标位置地址 | `0x03` | 论文 + `SVHConstants.cs` | 可作为当前主假设 |
| checksum 形式 | `CHECK1=sum`, `CHECK2=xor` preview | `SVHSerialInterface.cs` | 可作为联调前参考默认值，待真机确认 |
| 链路思路 | `TCP/IP -> 串口服务器 -> RS485 -> SVH` | 论文 + Unity/C# | 可作为联调前链路假设 |
| 9 通道名称与顺序 | 已与 Unity/C# 参考实现对齐 | `SVHConstants.SVHChannel` | 可作为 preview 和真机前校准的首选假设顺序，不能替代实测确认 |

### 关于 `40 / 64` 的歧义

这里必须单独拆开说，因为论文和本地 C# 代码之间存在“可以有两种读法”的地方。

论文里的文字更像是在说:

- “发送到控制器的命令包长度为 40 字节”
- “返回的包的长度为 64 字节”

这两句里的“包长度”读起来更像整包长度，而不是 payload 长度。

但本地 Unity/C# 代码里又能看到:

- `SVHSerialSendPacket.data` 固定为 `40` 字节
- `SVHSerialReceivePacket.data` 固定为 `64` 字节
- struct 里另外还有:
  - `head1`
  - `head2`
  - `index`
  - `address`
  - `length`
  - `check_sum1`
  - `check_sum2`

因此，当前 baseline 在 [svh_protocol.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/svh/svh_protocol.py) 里采用的是下面这个“基于本地 C# 代码推断出的 framing 假设”:

- command payload: `40`
- command frame: `48`
- response payload: `64`
- response frame: `72`

这个拆法目前只能表述为:

- 这是根据本地 Unity/C# 代码推断出的 framing 假设
- 与论文文字表述存在歧义
- 真机前必须通过真实收发进一步确认

还要再补一句:

- 如果 `48 / 72` 的理解部分来自结构体字段布局推断，那么还必须确认是否存在 struct packing、字段对齐方式或序列化实现细节带来的隐藏差异

不能写成“这已经是最终官方正确 framing”。

## 4. 9 通道顺序假设

当前仓库在 `svh_9ch` 模式下使用的通道顺序是:

1. `thumb_flexion`
2. `thumb_opposition`
3. `index_finger_distal`
4. `index_finger_proximal`
5. `middle_finger_distal`
6. `middle_finger_proximal`
7. `ring_finger`
8. `pinky`
9. `finger_spread`

这些名字和顺序来自:

- `D:/SVH/RoboticArm/Assets/Scripts/SVH/SVHConstants.cs`

更准确的表述应该是:

- 当前 9 通道顺序已与 Unity/C# 参考实现对齐
- 可作为预览层和真机前校准的首选假设顺序
- 但不能替代设备实测顺序确认

还需要额外强调:

- 这些名称是当前软件层的 preview 命名
- 不等于已经证明每个通道都与一个独立的人手式物理关节一一对应
- 对于 `9 驱动 / 20 关节` 这类体系，通道命名、机械耦合和最终运动表现之间仍可能存在非一一对应关系

也就是说，这套顺序现在足够适合:

- `svh preview`
- `target_ticks_preview`
- 联调前参数讨论

但还不够资格当成“真实设备顺序已最终确认”。

## 5. ticks 语义声明

当前 [svh_layout.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/svh/svh_layout.py) 和 [svh_9ch_preview.yaml](/D:/VR/HandAi/single-hand-teleop-baseline/configs/svh_9ch_preview.yaml) 里的:

- `svh_9ch_open_ticks`
- `svh_9ch_closed_ticks`

它们的语义必须说清楚:

- 这些值当前只是 preview 标尺
- 它们来源于 Unity/C# 参考实现里的 home-setting 范围风格
- 它们不代表已经绑定到真实 homing 零位的绝对编码器命令值
- 它们不代表已经通过真实 SVH 设备验证的安全上下限
- 它们也不代表“只要发出去就一定是对的”

换句话说，当前这些 ticks 更准确的理解是:

- 软件 preview 映射使用的参考终点

而不是:

- 真机可直接下发的最终设备量纲

尤其还没确认的东西包括:

- 这些 ticks 是不是绝对编码器值
- 它们是不是相对 homing 零位定义
- 同一通道正负方向到底对应哪一个机械方向
- open/closed 到底是视觉语义上的 open/closed，还是设备安全范围内的 preview open/closed

## 6. 9 通道 ticks 参考表

当前 preview ticks 参考值定义在:

- [svh_layout.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/svh/svh_layout.py)
- [svh_9ch_preview.yaml](/D:/VR/HandAi/single-hand-teleop-baseline/configs/svh_9ch_preview.yaml)

| Ch | Preview 名称 | 当前 open ticks | 当前 closed ticks | 现阶段更准确的理解 | 真机前必须继续确认 |
| --- | --- | --- | --- | --- | --- |
| 0 | `thumb_flexion` | `-5000` | `-175000` | preview 终点，不是已绑定零位的最终命令 | 编码器方向、零位、机械极限、闭合安全上限 |
| 1 | `thumb_opposition` | `-5000` | `-150000` | preview 终点，不是最终对掌标定值 | opposition 方向、open 基线、pinch 时实际对掌范围 |
| 2 | `index_finger_distal` | `-2000` | `-47000` | preview 终点，不是最终 distal 命令区间 | 负方向是否正确、pinch 区间是否需要单独收窄 |
| 3 | `index_finger_proximal` | `2000` | `42000` | preview 终点，不是最终 proximal 命令区间 | 正方向是否正确、home 是否确实落在正区间 |
| 4 | `middle_finger_distal` | `-2000` | `-47000` | preview 终点，不是最终 distal 命令区间 | middle distal 的符号、限位、与 index distal 是否可共用范围 |
| 5 | `middle_finger_proximal` | `2000` | `42000` | preview 终点，不是最终 proximal 命令区间 | middle proximal 的符号、限位、与 index proximal 是否可共用范围 |
| 6 | `ring_finger` | `-2000` | `-47000` | preview 终点，不是最终 ring 命令区间 | ring 单通道安全范围和闭合余量 |
| 7 | `pinky` | `-2000` | `-47000` | preview 终点，不是最终 pinky 命令区间 | pinky 单通道安全范围和闭合余量 |
| 8 | `finger_spread` | `-2000` | `-47000` | preview 终点，不是最终 spread 命令区间 | spread 真实方向、open/grasp/pinch 三种 spread 策略是否需要独立标定 |

## 7. 当前 preview 映射公式与优先级

这部分是后面真机校准时最容易“口头理解不一致”的地方，所以这里直接按当前代码写清楚。

代码入口在:

- [svh_adapter.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/svh/svh_adapter.py)

### 7.1 总体流程

当前 `svh_9ch` 模式下，命令的合成顺序是:

1. 先根据 `control_representation` 或 `gesture_fallback` 生成每个通道的 `alpha`
2. 再把 `alpha` 线性插值成归一化 `target_positions`
3. 再把 `target_positions` 线性插值成 `target_ticks_preview`

也就是:

```text
alpha -> normalized position -> preview ticks
```

具体到代码里:

```text
position = lerp(svh_position_open_value, svh_position_closed_value, alpha)
tick = round(open_tick + alpha * (closed_tick - open_tick))
```

因此当前优先级是:

- scale 先作用在 `alpha`
- `open_ticks / closed_ticks` 在最后一层负责把 `alpha` 投影到 ticks

不是:

- 先按 ticks 缩放
- 再回推 positions

### 7.2 `open_ticks / closed_ticks` 和 scale 的关系

当前关系是:

- `svh_9ch_open_ticks / svh_9ch_closed_ticks` 定义每个通道的 ticks 两端
- `svh_thumb_opposition_scale`, `svh_open_spread_scale`, `svh_grasp_spread_scale`, `svh_pinch_spread_scale` 这些 scale 只影响 `alpha`
- 然后 `alpha` 才被换算成 positions 和 ticks

所以更准确的说法是:

- 先算姿态强度
- 再决定归一化位置
- 最后才落到 ticks

### 7.3 `grasp` 家族的当前公式

当前 `svh_9ch` 的 `grasp` 映射是:

- `thumb_close = 0.75 * (grasp_close * svh_thumb_grasp_scale) + 0.25 * thumb_flex`
- `thumb_opp = max(thumb_flex, grasp_close * svh_thumb_opposition_scale)`
- `spread_alpha = (1 - grasp_close) * svh_open_spread_scale + grasp_close * svh_grasp_spread_scale`

其余手指的当前写法是“整体抓握强度”和“单指 flex”做加权混合:

- index distal: `0.65 * grasp_close + 0.35 * index_flex`
- index proximal: `0.80 * grasp_close + 0.20 * index_flex`
- middle distal: `0.65 * grasp_close + 0.35 * middle_flex`
- middle proximal: `0.80 * grasp_close + 0.20 * middle_flex`
- ring: `0.75 * grasp_close + 0.25 * ring_flex`
- pinky: `0.70 * grasp_close + 0.30 * little_flex`

这里的优先级可以理解为:

- `grasp_close` 是主导量
- 单指 `flex` 是微调量
- `finger_spread` 不是单独从某个手指 flex 来，而是从 `open` 到 `grasp` 两种 spread 策略之间插值得到

### 7.4 `pinch` 家族的当前公式

当前 `svh_9ch` 的 `pinch` 映射是:

- `support_value = max(support_flex, effective_pinch_strength * svh_pinch_support_scale)`
- `thumb_flexion = 0.85 * pinch_close + 0.15 * thumb_flex`
- `thumb_opposition = 0.75 * pinch_close + 0.15 * thumb_flex + 0.10 * svh_thumb_opposition_scale * grasp_close`
- `index_distal = 0.80 * pinch_close + 0.20 * index_flex`
- `index_proximal = 0.70 * pinch_close + 0.30 * index_flex`
- `finger_spread = pinch_close * svh_pinch_spread_scale`

中指、无名指、小指的当前策略是:

- 取 `max(单指当前 flex, support_value)`

也就是说:

- pinch 先优先闭合 thumb/index
- support fingers 只保持小幅辅助屈曲
- `finger_spread` 在 pinch 下是单独一条分支，不和 `grasp_spread_scale` 叠加

### 7.5 `open` 家族的当前公式

当前 `svh_9ch` 的 `open` 映射相对简单:

- 绝大部分通道 `alpha = 0`
- `finger_spread = svh_open_spread_scale`

所以现在 `open` 时 `spread` 是独立保留一点张开基线的，不是完全零。

### 7.6 `gesture fallback` 的当前优先级

当前只有满足下面条件时才允许 fallback:

- `svh_enable_gesture_fallback = true`
- `features_valid = false`
- `gesture in {open, fist, pinch}`

默认配置里这个开关是 `false`，所以更接近真机阶段时:

- 低质量帧
- 无效连续特征
- 过渡帧

都不会自动合成 fallback 命令。

## 8. 当前仓库里真正可调的校准旋钮

如果后面继续沿用这个 baseline，下表这些配置是最先需要按设备校准的入口。

| 配置项 | 当前作用 | 在公式里影响哪里 | 真机前怎么处理 |
| --- | --- | --- | --- |
| `svh_preview_layout` | 选择 `compact5` 或 `svh_9ch` | 选择映射分支 | 真机侧应优先基于 `svh_9ch` 校准 |
| `svh_9ch_open_ticks` | 9 通道 open 参考 ticks | ticks 终点 | 替换成设备实测 home/open 参考值 |
| `svh_9ch_closed_ticks` | 9 通道 closed 参考 ticks | ticks 终点 | 替换成设备实测安全闭合上限 |
| `svh_thumb_grasp_scale` | 调整 grasp 时拇指 flexion 强度 | `thumb_close` | 按 grasp 手型实测校准 |
| `svh_thumb_opposition_scale` | 调整拇指对掌量 | `thumb_opp`, 部分 pinch 对掌 | 按 open/grasp/pinch 三类手型实测校准 |
| `svh_pinch_support_scale` | 调整 pinch 时支撑手指参与度 | `support_value` | 按真实 pinch 稳定性校准 |
| `svh_open_spread_scale` | open 时 spread 基线 | `finger_spread` in open/grasp blend | 按张手姿态校准 |
| `svh_grasp_spread_scale` | grasp 时 spread 基线 | `finger_spread` in grasp blend | 按抓握姿态校准 |
| `svh_pinch_spread_scale` | pinch 时 spread 基线 | `finger_spread` in pinch | 按 pinch 姿态校准 |
| `svh_protocol_sync_bytes` | preview sync header | packet header | 目前可先保持不变 |
| `svh_protocol_use_little_endian` | preview 字节序 | packet packing | 必须通过真实收发确认 |
| `svh_enable_gesture_fallback` | continuous features 无效时是否允许 fallback | fallback 分支开关 | 真机阶段建议继续保持 `false` |

## 9. Packet / 协议层还要继续校准的部分

当前 [svh_protocol.py](/D:/VR/HandAi/single-hand-teleop-baseline/src/svh/svh_protocol.py) 只是更贴近论文和 Unity/C# 驱动的 skeleton。下面这些在真机前必须继续确认:

| 项目 | 当前状态 | 真机前必须确认 |
| --- | --- | --- |
| `0x09 SetControlState` payload | 只把首字节写进 40-byte payload preview | 控制状态位含义、保留位、是否还有额外字段 |
| `0x03 SetControlCommand AllChannels` payload | 已按 9 个 `int32` preview 打包 | 真实字节序、真实通道顺序、padding 规则 |
| 地址高 4 位语义 | 只在注释里保留“可能用于 channel selector” | 单通道命令时高 4 位到底怎么编码 |
| `length` 字段 | 目前跟随“payload 40/64 + frame 48/72”假设 | 论文里的“包长度”到底指 payload 还是整帧 |
| `CHECK1 / CHECK2` | 已做 `sum/xor` preview | 计算范围是否只覆盖 payload，还是覆盖 address/length；sum 是否按 8-bit 截断/回绕；xor 是否覆盖同样范围 |
| `index` / packet sequence | baseline 还没做真实收发序号语义 | 是否要求递增、回环、重传相关规则 |
| response parsing | 还没有解析真实返回帧 | 状态、fault、enable mask、位置、电流、编码器值等真实结构 |
| controller feedback | 还没接收真实控制器反馈 | 哪些命令需要轮询，哪些需要状态缓存 |

## 10. 链路与时序还要继续校准的部分

就算 packet 字节对了，真实链路也还需要单独校准:

| 层级 | 当前状态 | 真机前必须确认 |
| --- | --- | --- |
| TCP 客户端 | 还没实现 | 连接、重连、超时、阻塞策略 |
| 串口服务器 | 只知道链路存在 | IP、端口、波特率、串口参数、缓存策略 |
| RS485 | 还没接 | 半双工时序、收发切换、设备地址冲突 |
| 初始化顺序 | 只有 preview `SetControlState` 概念 | 上电、清 fault、enable channels、homing 的真实顺序 |
| 发送节奏 | 现在只是摄像头帧驱动 preview | 真机刷新频率、去抖、节流、丢包策略 |
| 安全策略 | 现在默认更保守，不发 fallback | 真机仍需加速度/步进限制、急停、超界保护 |

## 11. 建议的责任分工

为了避免以后“大家都以为别人确认过了”，建议至少按下面三类角色拆责任:

| 角色 | 主要负责确认什么 |
| --- | --- |
| 视觉/映射侧 | `control_representation` 是否稳定、`open/fist/pinch` 映射是否符合操作意图、哪些 scale 该怎么调 |
| 协议/链路侧 | sync bytes、地址、长度、endianness、checksum、TCP/串口服务器/RS485 收发 |
| 硬件/标定侧 | homing、零位、正负方向、机械限位、安全闭合范围、各通道 open/closed ticks |
| 集成验收侧 | 单通道小步进、全通道联动、fault 恢复、无效帧时是否安全 |

## 12. 真机接入前的最小验收清单与通过判据

后面真开始接 SVH 时，建议至少按这个顺序验收，并且每一步都明确“什么叫通过”。为了避免“到底从哪里观察”的歧义，这里把观察来源也一起写出来。

| 步骤 | 验收内容 | 通过判据 | 主要观察来源 |
| --- | --- | --- |
| 1 | 只连链路，不发运动命令 | TCP 能稳定连接；串口服务器可达；没有立即断链；如果有 response，至少能收回可解析头部 | TCP 连接日志、串口服务器日志、原始 response header |
| 2 | 只测 `SetControlState` | controller 状态有可观测变化；enable/fault 清理符合预期；没有新增 fault | controller state 字段、fault/state bits、返回帧解析结果 |
| 3 | 只测单通道小步进 | 指定通道朝预期方向发生小位移；反馈变化与通道一致；不超界、不碰撞、不触发 fault | 肉眼可见位移、encoder/position feedback、fault/state bits |
| 4 | 校准 `open_ticks / closed_ticks` | 每个通道都完成 home/zero、方向、最小安全值、最大安全值确认；open/closed 姿态与意图一致 | homing 记录、单通道标定表、encoder feedback、人工观察 |
| 5 | 再测 `AllChannels` | 多通道同时响应；没有明显串扰；没有 fault；姿态组合和预期一致 | 全通道反馈、fault/state bits、人工观察、日志对照 |
| 6 | 最后接入视觉输出 | 无手/invalid 帧不发危险命令；`open/fist/pinch` 可稳定触发；过渡帧不会产生明显错误动作 | JSON/JSONL 日志、控制器反馈、人工观察、必要时录像回放 |

## 13. 当前最适合继续保留 preview 的部分

在进入真机前，下面这些设计值得继续保留:

- `control_representation`
  - 把视觉特征和硬件命令之间隔开
- `features_valid / command_ready`
  - 让“特征可算”和“命令可发”分层
- `svh_enable_gesture_fallback: false`
  - 更接近硬件阶段时更安全
- `svh_9ch` + `target_ticks_preview`
  - 足够像真实 SVH，又还没有假装真机已通

## 14. 现在还不能声称已经完成的部分

基于当前仓库状态，仍然不能声称:

- 已完成真实 SVH 协议实现
- 已完成真实 TCP/IP 到串口服务器的发送
- 已完成真实 RS485 联调
- 已完成实体手通道校准
- 当前 preview ticks 已可直接作为真机命令

更准确的说法应该是:

- 我们已经有一套更接近 SVH 的 `preview / skeleton`
- 并且已经把未来真机接入前需要继续校准的关键点、公式和验收判据拆清楚了
