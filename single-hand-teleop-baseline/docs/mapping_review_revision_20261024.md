# 2026-10-24 映射复审修订：R1、R2、S1、S4

本轮按工作区的 `HandAi_映射修复复审_20261024.md` 修改。R1 选择关闭 AI 控制 flex 的时间滤波，将设备限速交给 DS；R2 放宽拇指参考区间、降低指尖项权重，并单独校准拇指抓握引导。数值验收通过，全套 **421 passed，Ruff 通过**。没有提交、推送或合并，没有运行相机、Unity 或实体设备。

当前入口为 [configs/ai_camera_v10.yaml](../configs/ai_camera_v10.yaml)，版本 `svh9-camera-v10-low-lag-thumb`。原 v2/v4 兼容配置以及首轮 v5–v9 的参数和报告均保留，旧模型/缓存不能混入新语义。

## 1. 实际修改

### R1：关闭映射层滤波

当前配置明确 `control_flex_filter_mode=none`。控制 flex 逐帧从原始 curl 及四指端点归一化得到，不更新 One Euro 状态，不添加窗口平均、时间渐变或映射层速度限制。配置中保留的三个旧滤波频率/系数字段用于既有配置结构，在 `none` 模式下不生效。

首轮的 2 Hz、beta=0.1 慢滤波只在标明“仅用于复现”的 v9 配置中保留。现在不再以减少快速动作的跳变总数作为调参目标。实体限速、限位和新鲜度检查仍由 DS 按设备能力处理，AI 的归一化目标不代替设备策略。

### R2：输入降敏与拇指抓握端点

| 参数 | 首轮 v9 | 当前 v10 |
|---|---:|---:|
| 拇指指尖参考开端 | 1.02 | 0.98 |
| 拇指指尖参考闭端 | 0.866 | 0.60 |
| 参考区间宽度 | 0.154 | 0.38 |
| 拇指指尖项权重 | 0.70 | 0.25 |
| 拇指抓握引导开端/闭端 | 直接使用整手 `grasp_close` | 0.10 / 0.65 |
| 拇指屈曲：引导/自身 flex 权重 | 0.75 / 0.25 | 0.95 / 0.05 |
| 抓握屈曲/对掌上限 | 0.85 / 0.75 | 0.85 / 0.75 |

单纯放宽输入区间或降权，在开发对照中仍无法同时满足 V5 的不滤波抖动上限和握拳行程。原因是拇指两通道还消费既有的整手抓握引导。因此 R2 同时对这段**当帧配比**单独定端点；它不改四指握紧度本身，不增加对掌角测量或时间滤波。

当前拇指计算如下，其中 `d` 是二维指尖到掌心距离/掌长，`angle` 是已有的拇指关节角代理，`g` 是原四指握紧度：

```text
thumb_curl = max(angle, 0.25 × clip((0.98 − d) / 0.38))
u = clip((g − 0.10) / 0.55)
抓握 ch0 = min(0.85, 0.95 × 0.85 × u + 0.05 × thumb_flex)
抓握 ch1 = min(0.75, max(thumb_flex, 0.75 × u))
```

v10 没有控制滤波，拇指的 `thumb_flex` 等于当帧 `thumb_curl`。捏合端点保留原公式，仍按 M6 的同帧连续权重与抓握端点混合；捏合及混合结果不受抓握端点的上限约束。对掌和屈曲仍是配比代理，独立拇指动作的幅度不代表真实关节角。

870 个主要录像帧的 ch2–ch8 与 v9 关闭滤波时**逐帧完全一致**。本轮没有实现 M3–M5 的直接外展、直接对掌或独立 MCP 测量。

### S1：单一、明确的当前配置

`ai_camera_v10.yaml` 包含全部 112 个有效配置键，没有 `extends`，不再通过 m1→m2→m7→m6→m8 继承。README 与接口文档以它为当前入口。五个首轮 `ai_mapping_m*.yaml` 的参数未改，文件头明确“仅用于复现”及对应历史版本。

### S4：行尾

仅把 `src/pipeline.py`、`src/svh/mapping_contract.py` 的工作区 CRLF 统一为 LF，没有修改 Git 全局设置或提交文件。`git ls-files --eol` 对两者均显示 `i/lf w/lf`；当前未提交工作区相对 HEAD 的累计 diff 分别为 9 增/3 删、23 增/1 删，未产生整文件替换。原始检查见 [行尾与 diff 输出](../outputs/mapping_review_revision_20261024/after/line_endings.txt)。

## 2. R1 验收

本轮严格沿用复审的两份数据：10-23 V1 重放 690 帧、M3-A V5 180 帧。分组固定采用原记录的手势；重放时间使用复审脚本原本的 payload `timestamp`。因此旧 v9 慢滤波的张手残留在本轮为 37/264，而首轮按媒体源时间计算为 33/264；两种历史结果不混用，原报告不覆盖。

| 条件 | V1 重放 | V5 | 结果 |
|---|---|---|---|
| R1(a) ch0–ch8 相对本版不滤波的最佳相关滞后 | 九通道均 0 帧 | 九通道均 0 帧 | 通过 |
| R1(c) 五指尖最大相邻位移 <0.05 掌长的帧对 | 593 对 | 143 对 | 合计 736 对 |
| 上述帧对中，任一通道变化 >0.2 | 0 | 0 | 通过 |

R1(b) 的 0→1 控制 flex 阶跃，在 **30 fps 和 13 fps 下五指首帧响应均为 100%**，满足至少 80%。真实 `HandPipeline` 的食指 45°、70°、拇指三个 B 动作也在两种帧率下验证：从已预热张手流切换后，首个改变帧的九通道目标即等于该姿态的静态目标，其他长指变化小于 0.05。

“0 帧”仅表示本次映射没有新增时间滤波滞后；未测量曝光、MediaPipe 内部处理、队列、网络或设备总延迟。

## 3. R2 验收与 R1(d) 报告

稳态抖动沿用复审定义：原手势标签不变的连续三帧，计算九通道二阶差分绝对值的 p95。这个分组不等于人工确认静止。R2 使用**关闭滤波**的当前配置，与相同分组的 v4 比较：

| 录像 | v4 ch0 / ch1 抖动 p95 | v10 不滤波 ch0 / ch1 | 相对 v4 倍数 |
|---|---|---|---|
| V1 重放 | 0.011217 / 0.010949 | 0.013600 / 0.012518 | 1.212448 / 1.143308 |
| V5 | 0.010577 / 0.011388 | 0.003516 / 0.008818 | 0.332431 / 0.774289 |

四个比值均不高于 1.5。完整九通道 p95 见 [机器验收结果](../outputs/mapping_review_revision_20261024/after/acceptance.json)；四指因首轮 M8 的范围扩展，与 v4 的抖动幅度仍有差异，不能把拇指达标写成所有通道更平稳。

B 张手基准的 ch1 从首轮约 0.174 降为 **0.025771 <0.10**。原 B 刚性旋转、单指隔离、70° 不误捏合的几何用例保持；完整输出见第 5 节。

握拳“上限附近”在本轮按**至少达到各自上限的 90%，且不超过上限**核验：

| 录像固定握拳组 | ch0 中位数 / 上限0.85 | ch1 中位数 / 上限0.75 |
|---|---:|---:|
| V1 重放，366 帧 | 0.812375（95.6%） | 0.750000（100%） |
| V5，64 帧 | 0.816981（96.1%） | 0.750000（100%） |
| V1 短片，43 帧 | 0.809998（95.3%） | 0.750000（100%） |

两段主要录像的四指通道中位数仍为 1；V1 短片四指中位数为约 0.945–1。抓握上限与捏合端点区别也有回归测试。

张手残留和跳变总数如实保留：

| 固定原张手组 / 全序列 | v4 | 首轮 v9 慢滤波 | 当前 v10 无滤波 |
|---|---:|---:|---:|
| V1 至少一个前八通道 >0.1 | 6/264 | 37/264 | 26/264 |
| V1 至少一个前八通道 >0.3 | 0 | 2 | 0 |
| V5 至少一个前八通道 >0.1 | 1/107 | 21/107 | 5/107 |
| V5 至少一个前八通道 >0.3 | 0 | 8 | 0 |
| V1 任一通道单帧变化 >0.2 | 8 | 1 | 8 |
| V5 任一通道单帧变化 >0.2 | 2 | 3 | 5 |

新结果的 13 次大跳变均伴随检测指尖最大位移约 0.085–0.191 掌长，原始帧号和每次幅度在机器结果中。V5 总次数没有下降，拇指抓握端点改变也会改变快速动作的通道幅度；本轮只按复审要求核验低位移时无大跳，不能据此宣称所有剩余变化都已证明为真实运动或所有映射伪影已消除。张手组的 0.1 残留也仍高于 v4，待人工标签区分过渡和静止情况。

## 4. 兼容与范围

[旧路径完整对照](../outputs/mapping_review_revision_20261024/legacy_full_comparison.json) 再次将当前未提交实现与原已提交实现比较：v2/v4 在三份记录上共 **2100 帧完整 payload** 逐字段一致，仅排除运行耗时 `latency_ms`；默认特征提取再核对 **1050 帧**完全一致。两份旧配置的参数没有改动。

复审脚本里相对于现场原记录仍有 7 帧 `gesture_context` 差异，沿用复审已经确认的 HEAD/现场重放差异；这不等于当前实现相对于 HEAD 改了旧输出。当前实现与 HEAD 使用同一重放输入和时间时，上述完整对照没有差异。

当前三份记录共 1050 帧均通过现有 JSON Schema，保持原九通道顺序和值域。新增语义和拇指参数进入 v10 映射契约，旧版本不允许静默接受这些参数。没有改变旧报告、缓存或 checkpoint，没有计算指纹或冻结收据。

未覆盖相机、Unity Play、真实捏合召回、其他人员、z 噪声和设备停止结果。没有人工 GT，不作准确率、泛化或真机验收结论。

## 5. 改后原始输出与复现

- [复审脚本一：review_mapping_fixes 改后输出](../outputs/mapping_review_revision_20261024/after/review_mapping_fixes.txt)
- [复审脚本二：review_b_cases_v9 改后输出](../outputs/mapping_review_revision_20261024/after/review_b_cases_v9.txt)
- [R1/R2 数值验收输出](../outputs/mapping_review_revision_20261024/after/acceptance.txt) / [完整 JSON](../outputs/mapping_review_revision_20261024/after/acceptance.json)
- [全套 pytest：421 passed](../outputs/mapping_review_revision_20261024/after/pytest.txt) / [JUnit](../outputs/mapping_review_revision_20261024/after/pytest.xml)
- [Ruff：All checks passed](../outputs/mapping_review_revision_20261024/after/ruff.txt)
- [执行命令与退出码](../outputs/mapping_review_revision_20261024/after/commands.json)
- [上述输出合订](../outputs/mapping_review_revision_20261024/after/evidence.md)

复审脚本新增 `--config`，默认当前 v10；第二个脚本保留文件名里的 `v9` 以兼容复审第 5 节原命令，不代表仍测 v9。B 的刚性旋转几何、稳态二阶差分分组和最佳相关滞后算法未变；滞后扩展至九通道，增加低位移跳变统计。B 诊断读取实际参考，并区分真实 2D 距离与实际采用的距离，避免旧日志把新 3D 值标为 2D。

原复审脚本、B 几何副本及旧原始输出保存在 `outputs/mapping_review_revision_20261024/before/`；原 `outputs/mapping_fix_review_20261024/*.txt` 未覆盖，新增同目录 `*_after.txt` 便于定位。

Windows CMD：

```bat
cd /d "D:\VR\HandAi\single-hand-teleop-baseline"
set "HANDAI_PY=C:\Users\31948\miniconda3\envs\handai-intent-prediction\python.exe"
"%HANDAI_PY%" -X utf8 outputs\mapping_fix_review_20261024\review_mapping_fixes.py --config configs\ai_camera_v10.yaml
"%HANDAI_PY%" -X utf8 outputs\mapping_fix_review_20261024\review_b_cases_v9.py --config configs\ai_camera_v10.yaml
"%HANDAI_PY%" -X utf8 outputs\mapping_review_revision_20261024\verify_review_criteria.py
```

运行两个脚本、数值验收、全套 pytest 和 Ruff 并统一保存结果：

```bat
"%HANDAI_PY%" -X utf8 outputs\mapping_review_revision_20261024\run_revision_checks.py
```

每次测试创建新的专用 `--basetemp` 并关闭 pytest 缓存。Ruff 范围为 `src tests scripts experiments examples/use_ai_api.py`。新增测试见 `tests/test_mapping_review_revision.py`；原 v9 慢滤波测试保留用于历史配置复现，不代表当前入口启用该滤波。
