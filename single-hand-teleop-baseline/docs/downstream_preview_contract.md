# 下游 Preview Contract

这份文档写给要消费 baseline 输出的人，比如 Unity 虚拟手、socket 客户端、数据记录脚本，或者未来的 SVH 预览联调程序。

先记住一句话：

**下游可以稳定读取每一帧 payload 的形状，但不应该把每一帧都当成可执行命令。**

## 当前 contract 能保证什么

每一帧都会输出一份 canonical payload。即使当前没检测到手、扩展没启用，或者这一帧质量不适合控制，下面两个对象仍然会存在：

- `control_representation`
- `svh_preview`

它们会退化成安全的 invalid / disabled 状态，而不是从 payload 里消失。这样下游代码不用每次猜字段在不在。

检测成功时，`landmarks_2d` 和 `landmarks_3d` 都必须恰好包含 21 个点；未检测到右手时两者必须同时为空。contract 不再接受“检测成功但只有部分 landmark”的模糊状态。

## 当前 contract 不能保证什么

这份 baseline 不会输出可以直接下发给真实 SVH 硬件的安全命令。

尤其注意：

- `svh_preview.target_positions` 可以给 Unity / preview 消费。
- `svh_preview.target_ticks_preview` 只能当 preview 元数据。
- `svh_preview.protocol_hint` 只能当布局和协议假设说明。
- 真实硬件仍然需要单独完成 homing、限位、速率限制、急停、ACK、timeout、fault 处理和真实 packet 校验。

## 下游最小读取规则

如果你只想快速接一个 Unity / socket preview，建议按这个规则做：

```text
if payload.detected
and payload.control_ready
and payload.svh_preview.enabled
and payload.svh_preview.valid:
    consume payload.svh_preview.target_positions
else:
    keep last safe pose or do nothing
```

如果你只是做日志记录，可以直接记录整份 payload，不需要丢弃 invalid 帧。invalid 帧对分析“什么时候丢手、什么时候贴边、什么时候控制不可用”很有价值。

## 顶层字段怎么读

最建议下游优先读取这些字段：

| 字段 | 含义 | 下游建议 |
| --- | --- | --- |
| `frame_index` | 当前帧编号 | 用于排序、排查丢帧 |
| `timestamp` | 检测与右手筛选结束附近的兼容时间戳 | 用于旧日志兼容，不应解释为采集时刻 |
| `detected` | 是否检测到目标右手 | false 时不要产生新动作 |
| `handedness` | 当前手标签 | 当前主线应关注 `Right` |
| `gesture_raw` | 当前帧即时手势 | 可用于调试，不建议直接驱动 |
| `gesture_stable` | 去抖后的稳定手势 | 可用于 UI 显示和辅助判断 |
| `control_ready` | 是否有可消费控制输出 | 下游动作门控首选字段 |
| `control_representation` | 连续控制中间层 | 适合控制逻辑或调参 |
| `svh_preview` | SVH / Unity 预览层 | 适合虚拟手或 socket preview |
| `fps` | 当前处理帧率 | 性能诊断 |
| `latency_ms` | 旧版 Python 视觉与 preview 处理耗时 | 不含采集、序列化、UDP、Unity 或真机 |
| `timing` | 可选的阶段时戳对象 | 同机 Python/Unity 延迟诊断首选 |
| `prediction_diagnostics` | 可选的默认关闭影子预测对象 | 只出现在独立 prediction 结果文件，不是 Unity/真机命令 |

完整字段列表以 schema 为准：

- [../schemas/frame_payload.schema.json](../schemas/frame_payload.schema.json)
- [../src/output/frame_payload_contract.py](../src/output/frame_payload_contract.py)

### 可选 `timing` v1

新版本实时主循环会附带：

| 字段 | 边界 |
| --- | --- |
| `source_read_start_unix_ms` | 调用输入源 `read()` 前 |
| `source_read_end_unix_ms` | 输入源返回后 |
| `detection_end_unix_ms` | MediaPipe 检测和右手筛选后 |
| `baseline_end_unix_ms` | 特征、质量门控和稳定手势完成后 |
| `preview_end_unix_ms` | `control_representation` 与 `svh_preview` 完成后 |
| `payload_ready_unix_ms` | contract 规范化与发送准备前 |
| `udp_send_attempt_unix_ms` | UDP 明确启用时，紧邻发送尝试前；否则为 `null` |

旧 payload 没有 `timing` 仍然合法。`timing` 使用 Unix epoch 毫秒；只有 Python 和 Unity 位于同一台电脑，或者跨设备时钟已经同步时，才能直接相减。Unity 当前记录的是“接收”和“目标应用”时间，不代表画面已经渲染，也不代表实体手已经响应。

### 可选 `prediction_diagnostics` v1

该对象只在显式开启 `--prediction-shadow` 时出现。当前帧 Unity UDP 数据报已经发送后，
主线程只把 payload 非阻塞提交到后台 worker；完成结果写入
`latest_prediction_shadow.json` / `prediction_session_*.jsonl`。baseline 的
`latest_svh_9ch.json` / `session_*.jsonl` 与 UDP 均不含该对象。Unity 不应依赖、消费或转发它。

关键约束：

- `source_frame_index` 必须与顶层 `frame_index` 一致；
- `source_timestamp_unix_ms` 必须等于顶层 `timestamp * 1000`；
- timing v1 的字段集合保持不变，推理开始/结束时刻在本对象内使用同一 Unix 时钟；
- 只有 `status="predicted"` 时 `ready=true`，hold/raw/gated 才是完整 `[H, 9]`；
- 其他状态必须清空预测数组并给出 `fallback_reason`；
- 无论状态如何，都不得改写 `svh_preview`。

状态和运行说明见
[单右手 9 通道意图预测影子模式](intent_prediction_shadow_mode.md)。

## `control_representation`

这个对象是“视觉特征”和“具体硬件命令”之间的中间层。它不绑定 SVH，也不绑定 Unity。

推荐下游读取：

| 字段 | 范围 | 含义 |
| --- | --- | --- |
| `features_valid` | bool | 当前帧是否具备可计算的连续特征 |
| `command_ready` | bool | 当前帧是否适合下游消费 |
| `preferred_mapping` | `grasp` / `pinch` / null | 当前更适合走抓握还是捏合映射 |
| `grasp_close` | `[0, 1]` / null | 抓握闭合程度 |
| `thumb_index_proximity` | `[0, 1]` / null | 拇指和食指接近程度 |
| `effective_pinch_strength` | `[0, 1]` / null | 经过门控后的有效捏合强度 |
| `support_flex` | `[0, 1]` / null | 中指、无名指、小指的平均支撑弯曲 |
| `finger_flex` | object | 五根手指用于下游控制的弯曲程度；可能包含配置启用的张手释放校正 |

视觉侧原始几何量始终保存在顶层 `finger_curl`。当
`control_open_release_enabled=true` 且稳定手势为 `open` 时，控制层会根据
`hand_open_ratio` 在 `control_open_release_start_ratio` 到
`control_open_release_full_ratio` 之间连续衰减 `finger_flex`，以抵消单目视角下
拇指、小指的残余 curl；该校正不作用于 `pinch` 或其他手势上下文。

兼容字段：

| 字段 | 说明 |
| --- | --- |
| `valid` | `command_ready` 的兼容镜像 |
| `pinch_strength` | `effective_pinch_strength` 的兼容镜像 |

推荐新代码优先读 `command_ready` 和 `effective_pinch_strength`。

## `svh_preview`

这个对象是给 SVH / Unity 风格预览用的输出层。它会把连续控制表示转换成通道目标值。

推荐下游读取：

| 字段 | 含义 | 下游建议 |
| --- | --- | --- |
| `enabled` | 本次运行是否启用 SVH preview | false 时不要消费目标数组 |
| `valid` | 当前帧是否产出了可用 preview | false 时不要消费目标数组 |
| `command_source` | `control_representation` 或 `gesture_fallback` | 真机前优先信任 `control_representation` |
| `target_channels` | 通道索引列表 | 与 `target_positions` 一一对应 |
| `target_positions` | 归一化 preview 目标，范围 `[0, 1]` | Unity / socket preview 首选消费值 |
| `target_ticks_preview` | preview-only ticks | 只用于调试和规划，不可当真机命令 |
| `protocol_hint` | 布局、单位、协议假设 | 用于下游解释字段 |

### `compact5` 和 `svh_9ch`

当前有两类 layout：

| layout | 通道数 | 适合用途 |
| --- | --- | --- |
| `compact5` | 5 | 快速看五指大致开合 |
| `svh_9ch` | 9 | 更接近 SVH / Unity 参考实现 |

`svh_9ch` 的通道顺序是：

1. `thumb_flexion`
2. `thumb_opposition`
3. `index_finger_distal`
4. `index_finger_proximal`
5. `middle_finger_distal`
6. `middle_finger_proximal`
7. `ring_finger`
8. `pinky`
9. `finger_spread`

## 常见帧状态

### 没检测到右手

通常会看到：

```json
{
  "detected": false,
  "gesture_stable": "unknown",
  "control_ready": false,
  "svh_preview": {
    "enabled": true,
    "valid": false,
    "target_channels": [],
    "target_positions": []
  }
}
```

下游建议：不要产生新动作。Unity 可以保持上一安全姿态，日志端可以照常记录。

### 检测到了手，但画面质量不适合控制

可能原因：

- 手太贴近画面边缘
- 大量 landmark 越界
- 掌心核心点不稳定

通常结果：

- `detected=true`
- `control_ready=false`
- `control_representation.command_ready=false`
- `svh_preview.valid=false`

下游建议：不要用这一帧更新动作。

### 正常可消费帧

通常结果：

- `detected=true`
- `control_ready=true`
- `control_representation.command_ready=true`
- `svh_preview.enabled=true`
- `svh_preview.valid=true`
- `target_channels.length == target_positions.length`

下游建议：消费 `target_positions`，并按 `target_channels` 映射到自己的虚拟手或预览对象。

## 示例

下面是一个省略后的有效 pinch 帧：

```json
{
  "frame_index": 233,
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

更多完整样例见：

- [../examples/sample_output.json](../examples/sample_output.json)
- [../examples/sample_output_svh_9ch.json](../examples/sample_output_svh_9ch.json)
- [../examples/sample_prediction_diagnostics.json](../examples/sample_prediction_diagnostics.json)
- [../examples/sample_session.jsonl](../examples/sample_session.jsonl)

## 真实 SVH 接入前的安全缺口

在这些环节完成前，`svh_preview` 必须继续停留在 preview / mock 层：

1. transport ACK / retry / timeout
2. homing 和零位确认
3. 每个通道的硬限位和软限位
4. watchdog / heartbeat
5. fault reset 流程
6. 操作员急停路径
7. 命令速率限制和平滑
8. 真实 packet packing / checksum / response parsing
9. `target_ticks_preview` 与真实设备单位的实测对应关系

真机相关检查表见 [SVH 真机接入前校准表](svh_real_hardware_calibration_table.md)。
