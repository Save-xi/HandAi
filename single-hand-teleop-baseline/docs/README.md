# 文档入口

这个目录放的是 README 之后才需要读的第二层文档。

如果你只是想先跑通项目，先看根目录的 [README.md](../README.md)。  
如果你已经能跑通 baseline，接下来按下面的顺序看会更顺。

## 推荐阅读顺序

1. [下游 Preview Contract](downstream_preview_contract.md)

   适合在这些时候读：

   - 你要写 Unity / socket / 日志消费端。
   - 你想知道 `control_representation` 和 `svh_preview` 哪些字段能直接用。
   - 你想判断一帧数据是否应该被下游消费。

2. [SVH 真机接入前校准表](svh_real_hardware_calibration_table.md)

   适合在这些时候读：

   - 你准备从 Unity 预览继续推进到真实 SVH。
   - 你需要确认协议、通道顺序、ticks、homing、限位、安全策略。
   - 你想列出真机联调前还不能声称已经完成的部分。

## 当前文档边界

这里的文档仍然服务于当前 baseline：

- 单右手视觉理解
- 冻结 frame payload
- 可选 `control_representation`
- 可选 `svh_preview`
- Unity / mock preview 联调

这里的文档还不是：

- 真实 SVH 官方协议文档
- 实体机械手安全控制说明书
- Unity 工程完整使用手册
- 双手或多手遥操作设计文档

## 一句话地图

```text
README.md
  -> 跑起来，知道项目是什么

docs/downstream_preview_contract.md
  -> 下游程序应该读哪些字段，怎么判断 valid / ready

docs/svh_real_hardware_calibration_table.md
  -> 真机接入前还要确认哪些协议、标定和安全项
```
