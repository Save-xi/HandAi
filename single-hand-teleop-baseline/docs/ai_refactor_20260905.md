# 2026-09-05 AI 工程重构记录

本轮以项目书第 3 节和第 4.1 节为 AI 路线，整理现有代码，不新增模型效果结论。

## 改动

- 提取 `HandPipeline` 和 `HandDetection`，实时处理、离线视频和设备关键点共用 AI 入口。
- 模型加载改为 `models/residual_motion4.json`，包含可读的参数、时序配置和权重路径；新增训练结果到模型配置的导出。
- 配置通过 `extends` 共用参数，默认 CLI 使用 `configs/ai.yaml`。
- 删除文件 SHA 核验、源码 AST 指纹、盲测冻结/收据流程，以及对应的行政性测试。
- 删除 Unity 源码快照、协议草稿、mock 传输和未使用的帧缓存；保留 JSON/JSONL、UDP 与预览数值接口。
- 旧实验/验收文档归档，数据集、checkpoint、机器报告和外部 Unity 工程保留。

Python/C# 代码净减少 **8,265 行**，包括被移除功能对应的测试。随机扰动场景仍使用原有的确定性种子生成方式，以保持数值比较条件；它不读取或核验文件身份。

## 实际验证

环境：`handai-intent-prediction`，Python 3.10；现有预测权重使用 CUDA。

| 检查 | 结果 |
|---|---|
| 修改前完整 pytest | 223 passed |
| 重构后完整 pytest | 171 passed |
| Ruff、compileall | 通过 |
| 15 个独立脚本的 `--help` | 全部通过 |
| 无设备关键点 API | 无需导入 MediaPipe/PyTorch 即可生成规范输出 |
| V5 前 180 帧与重构前代码比较 | 关键点、特征、手势、连续表示及 9 通道数值完全一致；计时与已删硬件 hint 不参与比较 |
| V1 全片媒体时间轴评测 | 690 帧；CUDA 预测与场景误差报告完成 |
| 旧报告导出新模型配置再推理 | 通过；现有权重未重训、未改写 |
| 最终运行主循环 | 90 帧，worker 提交/完成均为 90，队列丢弃为 0，正常退出 |
| 最终日志配对与计时顺序 | 90 帧逐帧配对；55 帧完成预测，其他为历史预热；预测与基础输出分离 |
| 第二轮独立训练 CLI 合成自检 | 训练、validation 选型、test 评测及报告生成完成 |

测试数量下降主要来自删除了盲测收据、快照和硬件协议功能。保留并运行了几何、手势、控制表示、模型兼容性、输出协议、断帧、后台预测、数据窗口、评测指标和数据切分测试。

本机详细输出位于 `outputs/ai_refactor_validation/`，包括 `pytest.txt`、`cli_smoke.json`、`prediction_smoke.json`、`runtime_pairing_summary.json`、`camera_eval/` 和 `training_smoke/`。

## 结果边界

这次验证确认重构后的 AI 路径可运行，并验证了指定视频片段的数值一致性。它没有重新评定模型泛化能力，也没有执行 Unity 或实体设备验收。现有模型仍为 9 通道代理预测，既有离线评测状态仍为 `offline_gate_passed=False`。合成训练自检的指标只说明代码流程可用。

后续算法工作见 [AI 路线图](ai_roadmap.md)。
