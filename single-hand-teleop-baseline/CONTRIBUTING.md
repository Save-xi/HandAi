# 贡献说明

这个子项目当前按“单右手 baseline”维护。

## 范围

- 保持 baseline 主路径稳定：输入 -> 检测 -> 特征 -> 手势 -> 可视化 / JSON。
- 将 `control_representation` 和 `svh_preview` 视为可选扩展。
- 不要把 Unity、真实 SVH 硬件或网络传输链路变成 baseline 的运行时前置依赖。

## 推荐环境

- Python `3.10`
- `conda activate single-right-hand-baseline`
  或者直接安装：
  `python -m pip install -r requirements.txt`

## 提交 PR 前

请在 `single-hand-teleop-baseline/` 目录下运行：

```bash
python -m compileall -q src
pytest -q
python src/main.py --help
```

如果本地安装了 CI 专用工具，也建议运行：

```bash
python -m ruff check src tests
```

## 代码风格

- 优先做小而明确的补丁，不要把无关重写混进来。
- payload 字段名要和冻结的 frame payload contract 保持一致。
- 只要行为发生变化，就补测试，尤其是手势逻辑、payload 导出、扩展降级相关部分。
- 新增可选特性时，必须保证 baseline-only 模式在没有硬件、Unity 和网络的情况下仍然可运行。
- 默认将注释、文档、日志文案、CLI 帮助等文字类内容写成中文；字段名、协议常量、外部接口名等兼容项保持英文。
