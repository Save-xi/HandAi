# 文档入口

这个目录放的是 README 之后才需要读的第二层文档。

如果你只是想先跑通项目，先看根目录的 [README.md](../README.md)。  
如果你已经能跑通 baseline，接下来按下面的顺序看会更顺。

## 推荐阅读顺序

1. [一阶段离线验收与风险清理](phase1_offline_acceptance.md)

   适合在这些时候读：

   - 当前没有可用摄像头，但要继续验证单右手 baseline。
   - 要运行全量 FreiHAND 验收并保存不可覆盖的 manifest。
   - 要用合成 9 通道轨迹检查 Unity，而不接真实硬件。

2. [单右手控制意图预测第一轮报告](intent_prediction_first_round_report.md)

   适合在这些时候读：

   - 要查看 H2O 跨受试者第一轮的正式指标、风险和证据边界。
   - 要决定下一轮使用 hold-last、GRU、TCN 还是 Transformer。
   - 要区分快速迭代结果和可进入论文表格的正式结果。

3. [单右手控制意图预测第二轮报告](intent_prediction_second_round_report.md)

   适合在这些时候读：

   - 要查看 residual GRU + validation 门控的正式离线结果。
   - 要核对 selection 先冻结、subject4 后加载的证据。
   - 要决定预测器是否可以进入默认关闭的影子模式。

4. [AI 部分与计划书口径对齐说明](ai_scope_and_proposal_alignment.md)

   适合在这些时候读：

   - 要向导师解释为什么当前使用 MediaPipe 单右手路线。
   - 要区分计划书中的算法/设备方向与当前已完成证据。
   - 要安排预测影子模式、延迟注入、量化和实验室联调的时间表。

5. [下游 Preview Contract](downstream_preview_contract.md)

   适合在这些时候读：

   - 你要写 Unity / socket / 日志消费端。
   - 你想知道 `control_representation` 和 `svh_preview` 哪些字段能直接用。
   - 你想判断一帧数据是否应该被下游消费。

6. [SVH 真机接入前校准表](svh_real_hardware_calibration_table.md)

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

docs/phase1_offline_acceptance.md
  -> 没有摄像头时怎样做离线验收、保存证据和检查 Unity

docs/intent_prediction_first_round_report.md
  -> H2O 第一轮预测结果、运动分层结论和下一轮模型选择

docs/intent_prediction_second_round_report.md
  -> residual GRU 验证集选型、冻结 test 结果和影子模式准入结论

docs/ai_scope_and_proposal_alignment.md
  -> 计划书算法/设备承诺、当前证据边界和八周 AI 任务接口

docs/svh_real_hardware_calibration_table.md
  -> 真机接入前还要确认哪些协议、标定和安全项
```
