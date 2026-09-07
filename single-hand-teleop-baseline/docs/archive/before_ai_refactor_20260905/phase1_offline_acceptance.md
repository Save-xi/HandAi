# 一阶段离线验收与风险清理

这套流程专门服务于当前条件：

- 主线仍是单右手 baseline。
- 当前没有可用摄像头。
- 不连接实体 SVH、AUBO 或其他硬件。
- 可以使用本机 FreiHAND 数据、合成 payload 和 Unity 本地 UDP。

## 已清理的边界风险

- `JsonExporter` 和默认配置都默认关闭 Unity UDP；只有明确配置或 `--send` 才发送。
- `requirements.txt` 与 `environment.yml` 使用同一套直接依赖。
- `detected=true` 时必须输出完整 21 个 2D/3D 点；`detected=false` 时两组点必须为空。
- 主循环导出前执行 normalize + validate，不再只做宽松规范化。
- 输出顺序固定为 UDP → 控制台 → JSON/JSONL，调试 I/O 不再排在 Unity 预览前。
- payload 可选携带 `timing v1`，Unity 可记录接收、排队和目标应用阶段诊断。
- FreiHAND 正式验收按 run 独立保存，不再覆盖固定 `reports/` 文件。

## 1. 最小离线检查

进入项目 Conda 环境后运行：

```powershell
python -m compileall -q src experiments/freihand_eval
python -m pytest -q
python scripts/send_unity_preview_demo.py --max-frames 5
```

第三条命令默认只生成并校验合成 payload，不发送 UDP。

## 2. 没有摄像头时检查 Unity

先在 Unity `Hnad.unity` 的 `RobotControlScript` Inspector 中确认：

- `Enable Baseline Udp Preview = true`
- `Baseline Udp Listen Port = 18080`
- `Apply Baseline Preview To Hardware = false`
- `Enable Legacy Gesture Snapping = false`

然后点击 Play，并运行：

```powershell
python scripts/send_unity_preview_demo.py --send
```

脚本会确定性地循环：

```text
open -> pinch -> open -> fist -> open
```

它不调用摄像头，也不运行 MediaPipe。它只证明：

- 规范 payload 能生成；
- Python UDP 能到达 Unity；
- Unity 的 9 通道展开和虚拟手目标应用能工作。

它不能证明视觉识别效果。默认拒绝向非本机地址发送；本阶段不要使用 `--allow-remote`。

## 3. 小样本 smoke run

先用少量数据检查整个验收工具：

```powershell
python experiments/freihand_eval/scripts/run_phase1_offline_acceptance.py `
  --config experiments/freihand_eval/configs/freihand_eval.yaml `
  --split evaluation `
  --max-samples 20
```

每次运行都会创建全新目录：

```text
experiments/freihand_eval/outputs/phase1_acceptance/<run_id>/
```

其中包含 manifest、配置快照、预测、指标、报告、SVG、命令日志和 SHA-256。

## 4. 全量一阶段验收

```powershell
python experiments/freihand_eval/scripts/run_phase1_offline_acceptance.py `
  --config experiments/freihand_eval/configs/freihand_eval.yaml `
  --split evaluation `
  --require-full-split
```

正式结果先看：

- `acceptance.md`
- `manifest.json`
- `metrics.json`

阶段回归下限在 `freihand_eval.yaml` 的 `phase1_acceptance` 中定义：

- 21 点完整率不低于 90%；
- 全 GT PCK@20px 不低于 80%；
- 有效预测 MPJPE 不高于 10 px；
- 检测器加手选择 P95 不高于 50 ms。

这些只是第一阶段回归下限，不是最终项目“准确率 90%”的替代表述。

## 5. Manifest 怎么看

- `status=completed`：命令完整执行。
- `checks`：每项自动门槛是否通过。
- `release_eligible=false`：通常表示工作树仍是 dirty、不是全量 evaluation、跳过测试或有门槛未通过。
- dirty 工作树的结果仍可用于开发对比，但不能当成正式冻结版本。

验收工具不会覆盖旧目录。运行中断时会保留 `.incomplete` 和失败原因。

## 当前仍无法在家验证的部分

- 实时视频连续性和快速运动；
- 无手负样本误检；
- 遮挡、强弱光和复杂背景；
- Unity 最终画面观感；
- 实体 SVH/AUBO 和 5G。

以后不一定要购买专用摄像头。普通手机录制固定测试视频后，可以通过：

```powershell
python src/main.py --config configs/unity_udp_preview.yaml `
  --video-file path/to/right_hand_test.mp4 `
  --headless `
  --save-jsonl
```

做确定性回放，再补齐真实视频风险。
