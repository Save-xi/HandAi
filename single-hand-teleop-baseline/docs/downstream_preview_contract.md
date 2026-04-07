# 下游 Preview Contract 说明

这份说明文档解释了：冻结后的 frame payload 中，哪些部分适合给 Unity / socket / SVH preview 类下游消费者读取。

## 1. Contract 范围

这个 baseline **不会**输出可以直接用于真实硬件的安全 SVH 命令。

它会输出两个对下游集成有价值的扩展对象，便于在真实硬件接入前先完成下游联调：

- `control_representation`
- `svh_preview`

这两个对象在 canonical payload 中始终存在，因此下游代码可以依赖稳定形状。当扩展被关闭，或当前帧还不适合产生命令时，对象仍然保留，只是退化成安全的 invalid / disabled 状态。

## 2. 推荐下游读取字段

### 顶层字段

对于 Unity、socket 或会话记录消费者，最建议优先读取这些字段：

- `frame_index`
- `timestamp`
- `detected`
- `handedness`
- `gesture_stable`
- `control_ready`
- `control_representation`
- `svh_preview`

### control_representation（控制表示）

推荐字段：

- `command_ready`
- `preferred_mapping`
- `grasp_close`
- `effective_pinch_strength`
- `finger_flex`
- `support_flex`

当前仍保留的兼容镜像字段：

- `valid`
  - `command_ready` 的兼容镜像
- `pinch_strength`
  - `effective_pinch_strength` 的兼容镜像

数值单位与范围：

- `grasp_close`：归一化 `[0, 1]`
- `thumb_index_proximity`：归一化 `[0, 1]`
- `effective_pinch_strength`：归一化 `[0, 1]`
- `support_flex`：归一化 `[0, 1]`
- `finger_flex.*`：归一化 `[0, 1]`

语义解释：

- `preferred_mapping="grasp"` 表示当前稳定手势上下文属于抓握类。
- `preferred_mapping="pinch"` 表示当前稳定手势上下文属于捏合类。
- `command_ready=false` 表示下游不应该把这一帧转换成新的控制动作。

### svh_preview（SVH 预览）

推荐字段：

- `enabled`
- `valid`
- `command_source`
- `target_channels`
- `target_positions`
- `target_ticks_preview`
- `protocol_hint`

数值单位与范围：

- `target_positions`
  - 归一化 preview 数值，范围在 `[0, 1]`
  - 推荐作为 Unity / socket preview 集成时的直接消费值
- `target_ticks_preview`
  - 仅用于 preview 的 encoder-like 整数
  - 适合 UI 调试和未来传输规划
  - **不是** 最终可直接下发给真实硬件的安全命令单位

关键语义：

- `enabled=false`
  - 本次运行没有启用 SVH preview 扩展
- `valid=false`
  - 当前帧没有可被下游消费的 preview 命令
- `command_source="control_representation"`
  - preview 来自稳定的连续控制输出
- `command_source="gesture_fallback"`
  - preview 来自演示导向的 fallback 逻辑

## 3. 示例 JSON 形状

```json
{
  "frame_index": 233,
  "timestamp": 1775039301.241,
  "detected": true,
  "gesture_stable": "pinch",
  "control_ready": true,
  "control_representation": {
    "command_ready": true,
    "preferred_mapping": "pinch",
    "grasp_close": 0.115,
    "effective_pinch_strength": 0.922,
    "finger_flex": {
      "thumb": 0.048,
      "index": 0.272,
      "middle": 0.022,
      "ring": 0.023,
      "little": 0.025
    }
  },
  "svh_preview": {
    "enabled": true,
    "valid": true,
    "command_source": "control_representation",
    "target_channels": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "target_positions": [0.791, 0.707, 0.792, 0.678, 0.185, 0.185, 0.185, 0.185, 0.092],
    "target_ticks_preview": [-139440, -107555, -37637, 28381, 6695, 6695, 6695, 6695, -6155],
    "protocol_hint": {
      "channel_layout": "svh_9ch",
      "position_units": "normalized_preview",
      "target_tick_units": "encoder_ticks_preview"
    }
  }
}
```

## 4. 下游可直接消费的内容

Unity / 虚拟手 preview 可以直接安全消费：

- `gesture_stable`
- `control_ready`
- `control_representation.grasp_close`
- `control_representation.effective_pinch_strength`
- `control_representation.finger_flex`
- `svh_preview.target_positions`
- `svh_preview.target_channels`

Socket / 日志客户端可以直接稳定记录：

- 整份冻结后的 payload
- `target_ticks_preview` 作为 preview 元数据
- `protocol_hint` 作为布局和单位说明

## 5. 仅限 Preview 的字段

这些字段更适合作为 preview / 调试元数据，而不是最终命令真值：

- `svh_preview.target_ticks_preview`
- `svh_preview.protocol_hint`
- `control_representation.valid`
- `control_representation.pinch_strength`

## 6. 真实 SVH 接入前的安全缺口

在真实灵巧手接入前，至少还缺少下面这些措施：

1. transport ACK / retry / timeout 处理
2. homing 和零位确认
3. 按通道的硬安全限位与标定
4. watchdog / heartbeat 与 fault reset 流程
5. 操作员急停路径
6. 命令速率限制与平滑策略
7. 确认 `target_ticks_preview` 与真实设备单位一致
8. 真实 packet packing / checksum / response parsing 验证

在这些环节完成前，`svh_preview` 必须继续保持在预览 / mock 层面。
