# 真实摄像头域盲测冻结协议 v1

- 状态：**blind_frozen**；仅适用于一次完整的 B1–B7 盲测。
- 适用范围：单右手视觉主链、`control_representation`、`svh_9ch` Unity 虚拟预览，以及默认关闭的 prediction shadow。
- 不适用：双手、多手、实体 SVH、串口/RS485、远端网络、实体关节误差、用户真实意图，或将 prediction 写入 Unity UDP。

本协议与 `experiments/intent_prediction/configs/camera_domain_blind_v1.json` 同时生效；JSON 是唯一的机器判定来源，本文说明其科学边界与采集动作。二者合并后，必须由 Git 跟踪树外（默认 ignored `outputs/`）的 blind-freeze manifest 记录 Git revision、配置/协议/运行时/模型和文件 SHA-256。为避免自引用，本协议和配置**不**预填自身 SHA 或 Git commit。

## 1. 结论边界

评测目标是同一条视频上、后续视觉映射得到的 `svh_preview`。它是视觉映射伪真值，不是实体关节真值，也不是用户主观意图真值。receiver tick 彼此重叠，不能作为独立统计样本；本盲测不产生显著性、临床/人因、真机安全、真实端到端延迟补偿或上线结论。

即使通过全部门，唯一允许的结论是：冻结的 `residual_motion4` 在这一次新录单右手视频的同步媒体时间轴重放中，相对 hold-last 维持了预注册的伪真值影子指标。baseline 仍是当前帧 `svh_preview`；prediction 继续 default-off、shadow-only，永不修改 Unity UDP payload，也不驱动实体 SVH。

## 2. 一次性冻结与文件身份

1. B1–B7 必须是开发视频 V1–V7 之后新录制的七个不同原始文件；不得剪辑、重编码、镜像翻转或用同一文件改名。
2. 评测前先计算七个 B 原始文件的 SHA-256。它们必须两两不同，且不得命中 JSON 中的 `forbidden_video_sha256`（开发 V1–V7 原始文件身份）；该禁止清单不是配置或协议自身的 SHA。
3. 合并已审查代码并录完 B1–B7 后，在 `HEAD == origin/main` 的最终合并 revision 上使用 Git 跟踪树外 manifest 一次性登记：Git revision/tree、配置、本文、runtime config、delay config、selection、checkpoint、原始 B 视频与环境身份。生成 manifest 和正式运行都要求 Git 完整工作树 clean（tracked 与非忽略 untracked 均为空）。程序用这些内容（不含文件路径、创建时间）计算确定性的 `attempt_token`；同一冻结身份即使改名或复制 manifest，token 也不变。
4. manifest 只能写到 ignored `experiments/intent_prediction/outputs/camera_domain_blind_freeze/manifests/<attempt_token>.json`；receipt 只能写到固定的 `attempt_receipts/<attempt_token>.json`。两个 CLI 都不提供 manifest/receipt 输出路径覆盖。正式运行显式传入 manifest、manifest 的外部预期 SHA-256，以及配置与本文的预期 SHA-256。
5. 正式解码前，程序再次核对原始 B 文件，并复制为 `sealed_inputs/<attempt_token>/` 下按 SHA-256 命名的只读副本；OpenCV 只打开这些副本。副本在解码后及算法后再次核对 SHA-256/字节数，避免“先 hash、后替换源路径”的时间差。
6. 输入门通过后、任何算法指标计算前，程序原子创建一次性 receipt；receipt 同时记录冻结身份与 baseline JSONL 的 SHA-256。输入门失败不创建 receipt，也不计算预测模型指标；一旦 receipt 已创建，`completed`、`failed` 和异常中断遗留的 `reserved` 都视为已经消耗，固定路径的存在会阻止重算。可捕获的 `KeyboardInterrupt`、`SystemExit` 与普通异常会尝试封存为 `failed`，回执写失败也不得掩盖原始异常。
7. 不得以“缺片段”“换镜像选项”或“不满意结果”为由替换已封存 B 文件、修改门或重新挑选 checkpoint。任何输入/时钟门失败进入 `input_or_clock_repair`，并不归因于模型。

### 2.1 唯一正式执行环境与命令形态

唯一环境是 Conda `handai-intent-prediction`。manifest 记录 Python 可执行文件路径和 SHA-256、Conda prefix/name、`conda list --explicit` 规范 SHA-256、`pip freeze --all` SHA-256、关键包版本及 CUDA 设备；正式运行逐项重算，不一致即在解码前失败。先录制七个原始视频到 `D:\HandAiVideos\camera_domain_blind_v1\`，再在**合并后的 clean revision** 上封存；冻结命令会加载冻结 predictor 并核对模型/环境身份，但不运行 MediaPipe 或预测回放。

```bat
cd /d D:\VR\HandAi\single-hand-teleop-baseline
conda activate handai-intent-prediction
python -X utf8 experiments\intent_prediction\scripts\freeze_camera_domain_blind.py ^
  --config experiments\intent_prediction\configs\camera_domain_blind_v1.json ^
  --video B1=D:\HandAiVideos\camera_domain_blind_v1\b1.mp4 ^
  --video B2=D:\HandAiVideos\camera_domain_blind_v1\b2.mp4 ^
  --video B3=D:\HandAiVideos\camera_domain_blind_v1\b3.mp4 ^
  --video B4=D:\HandAiVideos\camera_domain_blind_v1\b4.mp4 ^
  --video B5=D:\HandAiVideos\camera_domain_blind_v1\b5.mp4 ^
  --video B6=D:\HandAiVideos\camera_domain_blind_v1\b6.mp4 ^
  --video B7=D:\HandAiVideos\camera_domain_blind_v1\b7.mp4
```

脚本输出固定的 `manifest`、固定的 `attempt_receipt`、确定性 `attempt_token`、manifest SHA、配置 SHA、协议 SHA 与 Git revision。把这些值原样代入且只执行一次；不要移动或复制 manifest：

```bat
cd /d D:\VR\HandAi\single-hand-teleop-baseline
conda activate handai-intent-prediction
python -X utf8 experiments\intent_prediction\scripts\run_camera_domain_eval.py ^
  --role blind ^
  --config experiments\intent_prediction\configs\camera_domain_blind_v1.json ^
  --video B1=D:\HandAiVideos\camera_domain_blind_v1\b1.mp4 ^
  --video B2=D:\HandAiVideos\camera_domain_blind_v1\b2.mp4 ^
  --video B3=D:\HandAiVideos\camera_domain_blind_v1\b3.mp4 ^
  --video B4=D:\HandAiVideos\camera_domain_blind_v1\b4.mp4 ^
  --video B5=D:\HandAiVideos\camera_domain_blind_v1\b5.mp4 ^
  --video B6=D:\HandAiVideos\camera_domain_blind_v1\b6.mp4 ^
  --video B7=D:\HandAiVideos\camera_domain_blind_v1\b7.mp4 ^
  --expected-config-sha256 <evaluation_config_sha256> ^
  --expected-protocol-sha256 <protocol_sha256> ^
  --blind-freeze-manifest <manifest> ^
  --expected-blind-freeze-manifest-sha256 <manifest_sha256>
```

## 3. 统一媒体规格

- 输入镜像固定为 `input_mirrored=false`；采集软件必须保持同一方向，不得看结果后切换。
- 每段分辨率至少 1280×720；容器 nominal FPS 与媒体首末 PTS 计算的 duration-based FPS 都必须在 [29, 31] Hz。
- 解码帧数必须等于容器声明帧数；媒体时间戳严格递增；每一个解码帧都必须来自 `container_pts_ms`，故 PTS fallback 比例必须为 0。
- 评测按媒体时间轴同步重放，不使用处理 wall-clock 或 latest-only worker。`offline_processing_capacity` 不是源视频 FPS，也不是实时吞吐/时延结论。

## 4. B1–B7 采集任务与输入门

| ID | profile | 任务与时长 | machine input gate |
|---|---|---|---|
| B1 | clean | open → fist → open 慢速循环 ×3，20–25 s | ready/valid 各 ≥95%，连续 ready ≥3 s、单次 invalid ≤0.5 s；stable open/fist 各 ≥90 帧；机器确认完整顺序 ≥3 次 |
| B2 | clean | open → pinch → open 慢速循环 ×3，20–25 s | ready/valid 各 ≥95%，连续 ready ≥3 s、单次 invalid ≤0.5 s；stable open/pinch 各 ≥90 帧；机器确认完整顺序 ≥3 次 |
| B3 | clean | 半握 → 全握 → 半握连续渐变 ×3，20–25 s | ready/valid 各 ≥95%，连续 ready ≥5 s、单次 invalid ≤0.5 s；`grasp_close` 范围 ≥0.25，low≤0.75/high≥0.90，low-high-low ≥3 次 |
| B4 | clean | 固定机位快速 open ↔ fist，15–20 s | ready/valid 各 ≥95%，连续 ready ≥3 s、单次 invalid ≤0.5 s；open-fist-open ≥5 次、转场 ≥10 次、转场中位间隔 ≤1.5 s |
| B5 | intentional-invalid | 有意遮挡一次；总长 20–25 s | 全局 ready/valid 都在 [40%, 85%]；0–5.5 s ready ≥95%，5.5–10.5 s invalid ≥90%，10.5–15 s recovery-ready ≥90%；恰好 1 个 3–7 s 长失效段并恢复 |
| B6 | intentional-invalid | 右手出画 → 入画 ×3；总长 21.5–25 s | 全局 ready/valid 都在 [40%, 85%]；三个 exit/reentry 窗逐一通过；恰好 3 个 1.8–4 s 长失效段且恢复 ≥3 次 |
| B7 | clean | 连续自然动作，24–32 s | ready/valid 各 ≥95%，连续 ready ≥3 s、单次 invalid ≤0.5 s；stable open/fist/pinch ≥45/45/30 帧，手势转场 ≥6 次且中位间隔 ≤3 s |

clean 的短暂失效不被掩盖；超过 0.5 s 即判输入不健康。文件名和手势总帧数都不能代替任务执行：B1/B2/B4 还核验有序手势 run，B3 核验连续 `grasp_close` 高低往返，B7 核验转场数。B5/B6 的失效是预定刺激，不能错误套用 clean 的全片 95% 门；它们同时核验固定窗口、长失效 episode 数、恢复次数和首尾 ready。invalid 窗口必须 `control_ready=false` 且 `svh_preview.valid=false`，recovery 窗口必须二者均为 true。

无效窗口仅证明单右手 preview contract 的无效/恢复状态。它不证明实体设备的安全张开；实体 SVH 仍必须独立完成协议、标定、限位、watchdog、急停、断线和硬件验收。

## 5. 预注册算法门

以下指标均为 50/100 ms primary delay 的 gated 相对 hold-last 改善；超过门槛不是把预测接入控制的授权。

| 类别 | 机器门 |
|---|---:|
| 总体 gated RMSE improvement | ≥ +3.0% |
| 总体 dynamic-q90 gated RMSE improvement | ≥ +5.0% |
| 总体 conditional prediction availability | ≥ 0.90 |
| 总体 end-to-end prediction coverage | ≥ 0.85 |
| 总体 gated P95 improvement | ≥ 0.0%（不得回退） |
| 每视频 gated RMSE improvement | ≥ -1.0% |
| 每视频 gated P95 improvement | ≥ -1.0% |
| 每视频可评估源行数 | ≥ 300 |
| gated `range_violation_rate` | = 0 |

dynamic-q90 只在足够动态样本的聚合轨迹上判断；不得因某一段没有定义该分桶而事后改变其定义。范围越界为零是硬安全边界，不以平均改善抵消。

## 6. 三分支与报告用语

- 输入、PTS、媒体或 B5/B6 任务窗未过门：`input_or_clock_repair`。不评价 v2，也不将问题写作算法退化。
- 输入通过但任一算法门失败：`v3_pre_registered_candidate`。只允许另立一次独立、预注册的 v3 实验；不覆盖或追调当前 v2。
- 全部通过：`keep_v2_shadow`。保持 default-off shadow；v3 至多是可选消融。

报告必须保留 `camera_domain_pseudo_ground_truth_shadow_only` 的 claim status，并明确 `udp_created=false`、`unity_payload_modified_by_prediction=false`、`real_svh_in_scope=false`。不得把本协议输出改写为真实意图预测、实体关节误差、真机可靠性或延迟补偿有效性。

## 7. 预先承认的脆弱性

- 开发集的总体 conditional coverage 为约 90.8%，故 0.90 门余量不足一个百分点；这是最脆弱的总体门之一。
- 开发集中最小逐视频 RMSE 改善约 +2.21%，最小逐视频 P95 改善约 +2.49%；每视频 -1% 非回退门有余量，但仍只由有限、相关的视频段支撑。
- B5 开发采集曾出现两个长无效段，不能用事后描述替代本协议中的固定时间窗；B6 的三个失效—恢复窗也必须逐一核验。
- 所有数值仅来自已有开发集用于**事前**定门。B1–B7 的正式结果不得再反向调门、调窗口、调模型或换文件。
