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

说明：

- `Log Baseline Preview Packets` 默认应该关掉，不然 Unity Console 会每帧刷日志，影响观察。
- `Enable Legacy Gesture Snapping` 默认必须关掉。它是旧的离散模板吸附逻辑，会把连续动作拉回固定姿态，干扰当前预览链。

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

## 一句话总结

如果你下次又忘了，就记住：

先开 `Hnad.unity`，确认 `UDP 预览开 / legacy snapping 关 / 端口 18080`，再跑：

```bash
python src/main.py --config configs/unity_udp_preview.yaml
```
