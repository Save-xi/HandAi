from __future__ import annotations

"""GRU、persistence 残差 GRU、TCN 与 Transformer 时序预测模型。"""

from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as functional
except ModuleNotFoundError:  # baseline 环境不强制安装 PyTorch。
    torch = None
    nn = None
    functional = None


TORCH_AVAILABLE = torch is not None


if TORCH_AVAILABLE:

    class GRUForecaster(nn.Module):
        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int,
            layers: int,
            dropout: float,
            horizon_count: int,
            output_size: int,
        ) -> None:
            super().__init__()
            self.horizon_count = horizon_count
            self.output_size = output_size
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, horizon_count * output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            sequence, _ = self.gru(x)
            logits = self.head(sequence[:, -1, :])
            return torch.sigmoid(logits.reshape(-1, self.horizon_count, self.output_size))


    class ResidualGRUForecaster(nn.Module):
        """从 hold-last 出发预测残差；默认保留旧通道约束，关键点可用无界坐标。"""

        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int,
            layers: int,
            dropout: float,
            horizon_count: int,
            output_size: int,
            max_delta: float | None,
            output_bounds: tuple[float, float] | None = (0.0, 1.0),
            external_residual: bool = False,
        ) -> None:
            super().__init__()
            if input_size != output_size and not external_residual:
                raise ValueError("残差模型需要同维输入输出；不同表示必须另行提供残差参考")
            if max_delta is not None and max_delta <= 0.0:
                raise ValueError("residual_gru.max_delta 必须为正数")
            self.horizon_count = horizon_count
            self.output_size = output_size
            self.max_delta = None if max_delta is None else float(max_delta)
            self.output_bounds = output_bounds
            self.external_residual = external_residual
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, horizon_count * output_size),
            )
            final = self.head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        def forward(self, x: torch.Tensor, residual_base: torch.Tensor | None = None) -> torch.Tensor:
            sequence, _ = self.gru(x)
            logits = self.head(sequence[:, -1, :]).reshape(-1, self.horizon_count, self.output_size)
            residual = logits if self.max_delta is None else self.max_delta * torch.tanh(logits)
            if self.external_residual:
                if residual_base is None or residual_base.shape != (len(x), self.output_size):
                    raise ValueError("跨表示残差必须显式提供 [N,output_size] 当前参考")
                hold_last = residual_base[:, None, :].expand(-1, self.horizon_count, -1)
            else:
                if residual_base is not None:
                    raise ValueError("当前模型没有启用外部残差参考")
                hold_last = x[:, -1:, :].expand(-1, self.horizon_count, -1)
            prediction = hold_last + residual
            return prediction if self.output_bounds is None else torch.clamp(prediction, *self.output_bounds)


    class CausalTemporalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.GELU()
            self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
            self.norm = nn.GroupNorm(1, out_channels)

        def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
            return conv(functional.pad(x, (self.left_padding, 0)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.residual(x)
            y = self.dropout(self.activation(self._causal_conv(x, self.conv1)))
            y = self.dropout(self.activation(self._causal_conv(y, self.conv2)))
            return self.activation(self.norm(y + residual))


    class TCNForecaster(nn.Module):
        def __init__(
            self,
            *,
            input_size: int,
            channels: list[int],
            kernel_size: int,
            dropout: float,
            horizon_count: int,
            output_size: int,
        ) -> None:
            super().__init__()
            if not channels:
                raise ValueError("TCN channels 不能为空")
            blocks = []
            current = input_size
            for index, channel_count in enumerate(channels):
                blocks.append(
                    CausalTemporalBlock(
                        current,
                        int(channel_count),
                        kernel_size=kernel_size,
                        dilation=2**index,
                        dropout=dropout,
                    )
                )
                current = int(channel_count)
            self.network = nn.Sequential(*blocks)
            self.horizon_count = horizon_count
            self.output_size = output_size
            self.head = nn.Sequential(
                nn.Linear(current, current),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(current, horizon_count * output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.network(x.transpose(1, 2))[:, :, -1]
            logits = self.head(features)
            return torch.sigmoid(logits.reshape(-1, self.horizon_count, self.output_size))


    class TransformerForecaster(nn.Module):
        def __init__(
            self,
            *,
            input_size: int,
            history_frames: int,
            d_model: int,
            nhead: int,
            layers: int,
            dim_feedforward: int,
            dropout: float,
            horizon_count: int,
            output_size: int,
        ) -> None:
            super().__init__()
            if d_model % nhead != 0:
                raise ValueError("Transformer d_model 必须能被 nhead 整除")
            self.input_projection = nn.Linear(input_size, d_model)
            self.position = nn.Parameter(torch.zeros(1, history_frames, d_model))
            nn.init.trunc_normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=layers,
                norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False,
            )
            self.horizon_count = horizon_count
            self.output_size = output_size
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, horizon_count * output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] > self.position.shape[1]:
                raise ValueError("输入历史长度超过模型 position 参数长度")
            encoded = self.encoder(self.input_projection(x) + self.position[:, : x.shape[1], :])
            logits = self.head(encoded[:, -1, :])
            return torch.sigmoid(logits.reshape(-1, self.horizon_count, self.output_size))


def build_model(
    name: str,
    *,
    history_frames: int,
    horizon_count: int,
    architecture: dict[str, Any],
    input_size: int = 9,
    output_size: int = 9,
    output_bounds: tuple[float, float] | None = (0.0, 1.0),
    external_residual: bool = False,
) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("当前解释器没有 PyTorch；请使用 experiments/intent_prediction/environment.yml 创建独立环境")
    normalized = name.lower()
    if normalized != "residual_gru" and (input_size != 9 or output_size != 9 or output_bounds != (0.0, 1.0) or external_residual):
        raise ValueError("M1 只为 residual_gru 实现关键点输出；其他模型保留旧 9 通道行为")
    if normalized == "gru":
        config = architecture.get("gru", {})
        return GRUForecaster(
            input_size=9,
            hidden_size=int(config.get("hidden_size", 128)),
            layers=int(config.get("layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            horizon_count=horizon_count,
            output_size=9,
        )
    if normalized == "residual_gru":
        config = architecture.get("residual_gru", architecture.get("gru", {}))
        return ResidualGRUForecaster(
            input_size=input_size,
            hidden_size=int(config.get("hidden_size", 128)),
            layers=int(config.get("layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            horizon_count=horizon_count,
            output_size=output_size,
            max_delta=config.get("max_delta", 0.35),
            output_bounds=output_bounds,
            external_residual=external_residual,
        )
    if normalized == "tcn":
        config = architecture.get("tcn", {})
        return TCNForecaster(
            input_size=9,
            channels=[int(value) for value in config.get("channels", [64, 96, 128])],
            kernel_size=int(config.get("kernel_size", 3)),
            dropout=float(config.get("dropout", 0.1)),
            horizon_count=horizon_count,
            output_size=9,
        )
    if normalized == "transformer":
        config = architecture.get("transformer", {})
        return TransformerForecaster(
            input_size=9,
            history_frames=history_frames,
            d_model=int(config.get("d_model", 128)),
            nhead=int(config.get("nhead", 4)),
            layers=int(config.get("layers", 3)),
            dim_feedforward=int(config.get("dim_feedforward", 256)),
            dropout=float(config.get("dropout", 0.1)),
            horizon_count=horizon_count,
            output_size=9,
        )
    raise ValueError(f"未知神经网络模型：{name}")
