# M1 关键点实验

任务：`h2o_camera3d_right21`；H2O 单种子小样本开发实验，subject4 为历史复评。

| 切分 | 方法 | 时距 ms | 归一化 MPJPE（序列等权） | MPJPE mm（序列等权） | 完整目标覆盖率 |
|---|---|---:|---:|---:|---:|
| validation | hold_last | 50 | 0.087943 | 8.400154 | 99.41% |
| validation | hold_last | 100 | 0.171921 | 16.421500 | 99.02% |
| validation | hold_last | 150 | 0.250747 | 23.950786 | 98.53% |
| historical_reevaluation | hold_last | 50 | 0.071384 | 6.710844 | 99.70% |
| historical_reevaluation | hold_last | 100 | 0.138824 | 13.050873 | 99.30% |
| historical_reevaluation | hold_last | 150 | 0.200395 | 18.839206 | 98.81% |
| validation | linear | 50 | 0.037905 | 3.620571 | 99.41% |
| validation | linear | 100 | 0.088479 | 8.451327 | 99.02% |
| validation | linear | 150 | 0.153480 | 14.660110 | 98.53% |
| historical_reevaluation | linear | 50 | 0.033401 | 3.140069 | 99.70% |
| historical_reevaluation | linear | 100 | 0.075014 | 7.052077 | 99.30% |
| historical_reevaluation | linear | 150 | 0.128619 | 12.091519 | 98.81% |
| validation | residual_gru | 50 | 0.082172 | 7.848848 | 99.41% |
| validation | residual_gru | 100 | 0.163118 | 15.580657 | 99.02% |
| validation | residual_gru | 150 | 0.240104 | 22.934212 | 98.53% |
| historical_reevaluation | residual_gru | 50 | 0.068147 | 6.406548 | 99.70% |
| historical_reevaluation | residual_gru | 100 | 0.135069 | 12.697901 | 99.30% |
| historical_reevaluation | residual_gru | 150 | 0.198012 | 18.615110 | 98.81% |

验证集选择：`linear`；常速度窗口：2 帧。

本轮只完成 M1 接线和小样本实测；单种子结果不替代 M2 正式收益对照。
`model.json` 始终指向 residual GRU 实验候选，不自动启用到实时链路。
网络耗时仅含设备驻留输入的 forward；不包含检测、归一化、传输或输出处理。
