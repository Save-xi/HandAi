# AI 时序预测实验

这里保留 H2O 数据转换、预测模型训练、validation 选型、误差统计和摄像头域评测。每次实验生成独立结果目录，可以正常重复运行。

当前学习目标为未来的 9 通道 `svh_preview` 代理序列。项目书中的 21 点轨迹预测与模型轻量化按 [AI 路线图](../../docs/ai_roadmap.md) 继续推进。

## 环境与数据

```bat
conda activate handai-intent-prediction
cd /d D:\VR\HandAi\single-hand-teleop-baseline
```

以下命令均从这个子项目目录执行。已有数据无需重复下载：

- H2O：`D:\VR\HandAi\datasets\H2O`。
- 当前 v2 预处理数据：`D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release`。
- 摄像头开发视频：`D:\HandAiVideos\camera_domain_dev_v1\v1.mp4` 至 `v7.mp4`。

H2O 数据清单记录来源、分人切分、帧率、投影方式及明文映射参数。默认 train=subject1+2、validation=subject3、test=subject4；subject4 已被历史实验使用。

## 训练

先用合成小数据检查训练到报告的完整路径：

```bat
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --synthetic-smoke --output-root outputs\training_smoke
```

使用已有数据训练：

```bat
python -X utf8 experiments\intent_prediction\scripts\run_second_round.py --config experiments\intent_prediction\configs\h2o_second_round.json --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release --output-root experiments\intent_prediction\outputs\second_round
```

模型结构、训练参数、候选及评测阈值在 JSON 配置中。保留 train/validation/test 分离：用 validation 选择模型和门控，随后评测 test。选中神经网络时，结果目录同时生成 `model.json`，供运行时直接加载；若选择 hold-last，则没有神经网络配置需要导出。

既有第一轮 `run_first_round.py` 仍可比较 hold-last、线性、Kalman、GRU、TCN、Transformer。合成自检用于验证代码，不作为算法效果结论。

## 将已有训练结果导出给运行时

```bat
python -X utf8 scripts\export_prediction_model.py --report experiments\intent_prediction\reports\second_round\20260829T024400_143036Z_report.json --output models\my_prediction.json
python -X utf8 scripts\run_prediction_shadow_smoke.py --config configs\ai.yaml --model models\my_prediction.json
```

运行时只需模型 JSON 与 checkpoint；原始评测报告不再参与启动核验。现有默认模型是 `models/residual_motion4.json`，旧 checkpoint 原样使用。

## 真实视频评测

```bat
python -X utf8 experiments\intent_prediction\scripts\run_camera_domain_eval.py --video V1=D:\HandAiVideos\camera_domain_dev_v1\v1.mp4 --video V5=D:\HandAiVideos\camera_domain_dev_v1\v5.mp4
```

可用任意不重复 ID；加 `--skip-prediction` 可仅评测姿态处理。脚本按容器 PTS 构建时间轴，失效时用源帧率连续补齐；不会把机器处理速度当作视频帧率。生成 JSON/Markdown 报告、视频指标和预测场景 CSV。

预测结果与后续视觉映射的代理标签比较，报告 hold-last/raw/gated 误差及条件覆盖率、总体覆盖率。新视频的实验用途由数据划分决定，不再靠文件身份或一次性收据管理。

## 延迟、抖动和丢包

```bat
python -X utf8 experiments\intent_prediction\scripts\run_delay_injection.py --data-root D:\VR\HandAi\datasets\H2O\processed\cross_subject_v2_open_release
```

保留原有网络扰动矩阵、随机种子和误差口径，方便与既有结果比较。这里评估软件注入扰动下的算法效果；通信和设备联调由对应成员处理。

## 数据重新预处理

修改标签或投影方式时，使用 `scripts/preprocess_h2o.py` 指向实际解压目录和新的输出目录。不要覆盖已用模型的标签；新数据清单直接保存参数，便于知道模型学的是什么。详细参数可用 `--help` 查看。

历史机器结果继续保留在 `reports/` 和被 Git 忽略的 `outputs/`。早期报告中的哈希字段仅是历史记录。
