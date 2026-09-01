# Unity Phase 1.5 最小源码快照

这不是完整 Unity 工程，而是从本机实际联调工程 `D:\SVH\RoboticArm` 抽取的最小可恢复快照。
它只覆盖当前项目已经验收的链路：

`单右手 baseline -> svh_preview -> 127.0.0.1:18080 -> Unity 虚拟手预览`

快照不包含场景、模型、纹理、`Library/`、`Temp/`，也不代表真实 SVH 硬件控制已经完成。

## 为什么保留这份快照

原 Unity 工程当前不是 Git 仓库。若只提交 Python 代码，已经通过行为回归的 UDP 接收器、
安全闸门和 batch 验收脚本仍可能丢失或无意漂移。因此这里把最关键的源码、包清单和
Unity 编辑器版本与 Python 版本一起保存，并用 `snapshot_manifest.json` 固定 SHA-256。

## 包含内容

- `Assets/Scripts/RobotControlScript.cs`：本机 UDP 接收、payload 合法性/时序闸门、
  watchdog 安全张开、9 通道到虚拟手关节的展开，以及退出 Play 时写出的有界 timing
  P50/P95/max 摘要。摘要默认落到 `Application.persistentDataPath/HandAiDiagnostics/`，不进入 UDP。
- `Assets/Editor/BaselineUdpSafetyBatch.cs`：loopback、invalid、乱序、过期和 watchdog
  的 Unity batchmode 行为验收。
- `Packages/manifest.json`、`Packages/packages-lock.json`：已修复红色包错误后的依赖快照。
- `ProjectSettings/ProjectVersion.txt`：验收使用的 Unity `2020.3.49f1` 版本记录。

## 使用边界

1. 正常运行仍使用 `D:\SVH\RoboticArm`，不会从本目录直接启动 Unity。
2. 需要恢复时，先备份目标 Unity 工程，再按相对路径逐文件比较或复制；不要覆盖不同
   Unity 版本、不同场景或含有新改动的工程。
3. `tests/test_unity_snapshot.py` 总会校验快照自身哈希；若本机外部 Unity 工程存在，还会
   对比外部源码，及时暴露两边漂移。
4. 这份快照的安全结论只适用于本机虚拟预览，不能作为真机协议、限位、急停或通信验收。
5. 同目录 `.gitattributes` 会禁用快照载荷的换行归一化，保证 Windows/Linux checkout
   后仍按 manifest 做逐字节 SHA-256 校验。

动态验收标记：`PHASE15_UNITY_SAFETY_BATCH_PASS`。
