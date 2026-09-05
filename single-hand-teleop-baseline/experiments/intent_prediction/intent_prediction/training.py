from __future__ import annotations

"""PyTorch 时序模型的统一训练、早停、测试与单样本延迟测量。"""

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

# CUDA 10.2+ 的确定性矩阵运算需要在首次 cuBLAS 调用前设置该变量。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .models import TORCH_AVAILABLE, build_model
from .sequence_data import WindowSplit

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(requested: str) -> str:
    normalized = requested.lower()
    if normalized == "auto":
        return "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 无法使用 CUDA")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"未知 device：{requested}")
    return normalized


def _loader(split: WindowSplit, *, batch_size: int, shuffle: bool, seed: int) -> Any:
    dataset = TensorDataset(torch.from_numpy(split.x), torch.from_numpy(split.y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _mean_loss(model: Any, loader: Any, criterion: Any, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            loss = criterion(model(x_batch), y_batch)
            total += float(loss.item()) * len(x_batch)
            count += len(x_batch)
    return total / max(1, count)


def _motion_reference(split: WindowSplit, *, recent_frames: int, percentile: float) -> float:
    count = min(max(2, int(recent_frames)), split.x.shape[1])
    score = np.mean(np.abs(np.diff(split.x[:, -count:, :], axis=1)), axis=(1, 2))
    return max(float(np.percentile(score, percentile)), 1e-8)


def _training_loss(
    prediction: Any,
    target: Any,
    history: Any,
    *,
    loss_config: dict[str, Any],
    motion_reference: float,
) -> Any:
    beta = float(loss_config.get("smooth_l1_beta", 0.03))
    mse_weight = float(loss_config.get("mse_weight", 0.0))
    motion_weight = float(loss_config.get("motion_weight", 0.0))
    recent_frames = min(max(2, int(loss_config.get("motion_recent_frames", 8))), history.shape[1])
    max_motion_ratio = float(loss_config.get("max_motion_ratio", 3.0))
    element_loss = functional.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    if mse_weight > 0.0:
        element_loss = element_loss + mse_weight * (prediction - target) ** 2
    sample_loss = torch.mean(element_loss, dim=(1, 2))
    if motion_weight <= 0.0:
        return torch.mean(sample_loss)
    motion_score = torch.mean(
        torch.abs(history[:, -recent_frames + 1 :, :] - history[:, -recent_frames:-1, :]),
        dim=(1, 2),
    )
    ratio = torch.clamp(motion_score / motion_reference, min=0.0, max=max_motion_ratio)
    weight = 1.0 + motion_weight * ratio
    return torch.sum(sample_loss * weight) / torch.sum(weight)


def _predict(model: Any, split: WindowSplit, *, batch_size: int, device: str) -> np.ndarray:
    loader = _loader(split, batch_size=batch_size, shuffle=False, seed=0)
    chunks = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            chunks.append(model(x_batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def predict_neural_checkpoint(
    checkpoint_path: Path,
    *,
    split: WindowSplit,
    batch_size: int = 256,
    device: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """加载本实验 checkpoint 并按与正式评测相同的批量路径推理。"""

    if not TORCH_AVAILABLE:
        raise RuntimeError("当前解释器没有 PyTorch")
    resolved_device = _resolve_device(device)
    if resolved_device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["history_frames"]) != split.x.shape[1]:
        raise ValueError("checkpoint 历史帧数与评测窗口不一致")
    if int(checkpoint["horizon_count"]) != len(split.horizon_ms):
        raise ValueError("checkpoint 预测距离数量与评测窗口不一致")
    if "horizon_ms" in checkpoint and checkpoint["horizon_ms"] != list(split.horizon_ms):
        raise ValueError("checkpoint 预测时间距与评测窗口不一致")
    model = build_model(
        str(checkpoint["model_name"]),
        history_frames=int(checkpoint["history_frames"]),
        horizon_count=int(checkpoint["horizon_count"]),
        architecture=dict(checkpoint["architecture"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    prediction = _predict(model, split, batch_size=batch_size, device=resolved_device)
    latency = _latency(model, split, device=resolved_device)
    return prediction, {
        "checkpoint": str(checkpoint_path.resolve()),
        "model_name": str(checkpoint["model_name"]),
        "seed": int(checkpoint["seed"]),
        "device": resolved_device,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "latency_single_window": latency,
    }


def _latency(model: Any, split: WindowSplit, *, device: str, measurements: int = 100) -> dict[str, float]:
    count = min(max(1, measurements), len(split.x))
    samples = torch.from_numpy(split.x[:count]).to(device)
    model.eval()
    with torch.no_grad():
        for index in range(min(10, count)):
            _ = model(samples[index : index + 1])
        if device == "cuda":
            torch.cuda.synchronize()
        values = []
        for index in range(count):
            start = time.perf_counter()
            _ = model(samples[index : index + 1])
            if device == "cuda":
                torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples": float(count),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def train_neural_model(
    name: str,
    *,
    train: WindowSplit,
    val: WindowSplit,
    test: WindowSplit,
    architecture: dict[str, Any],
    training_config: dict[str, Any],
    checkpoint_path: Path,
    seed: int,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("当前解释器没有 PyTorch")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Transformer 默认可能选到非确定性的 memory-efficient attention。
        # 第一轮数据规模很小，固定使用 math SDP 更适合可复现比较。
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)

    device = _resolve_device(str(training_config.get("device", "auto")))
    model = build_model(
        name,
        history_frames=train.x.shape[1],
        horizon_count=train.y.shape[1],
        architecture=architecture,
    ).to(device)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = checkpoint_path.parent.parent / "training_progress.jsonl"
    batch_size = int(training_config.get("batch_size", 256))
    epochs = int(training_config.get("epochs", 40))
    patience = int(training_config.get("patience", 6))
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    gradient_clip = float(training_config.get("gradient_clip", 1.0))
    loss_config = dict(training_config.get("loss", {}))
    run_label = str(training_config.get("run_label", name))
    motion_reference = _motion_reference(
        train,
        recent_frames=int(loss_config.get("motion_recent_frames", 8)),
        percentile=float(loss_config.get("motion_reference_percentile", 75.0)),
    )

    train_loader = _loader(train, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = _loader(val, batch_size=batch_size, shuffle=False, seed=seed)
    criterion = nn.SmoothL1Loss(beta=float(loss_config.get("smooth_l1_beta", 0.03)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    with progress_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "model_started",
                    "model": name,
                    "run_label": run_label,
                    "device": device,
                    "seed": seed,
                    "loss": loss_config,
                    "motion_reference": motion_reference,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _training_loss(
                model(x_batch),
                y_batch,
                x_batch,
                loss_config=loss_config,
                motion_reference=motion_reference,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            running += float(loss.item()) * len(x_batch)
            seen += len(x_batch)
        train_loss = running / max(1, seen)
        val_loss = _mean_loss(model, val_loader, criterion, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        improved = val_loss < best_val - 1e-8
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                should_stop = True
            else:
                should_stop = False
        if improved:
            should_stop = False
        progress = {
            "event": "epoch_completed",
            "model": name,
            "run_label": run_label,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "improved": improved,
            "elapsed_seconds": time.perf_counter() - started,
        }
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(progress, ensure_ascii=False) + "\n")
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        if should_stop:
            break
    optimization_seconds = float(time.perf_counter() - started)

    if best_state is None:
        raise RuntimeError(f"{name} 未产生可用 checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save(
        {
            "model_name": name,
            "state_dict": best_state,
            "history_frames": int(train.x.shape[1]),
            "horizon_count": int(train.y.shape[1]),
            "horizon_ms": list(train.horizon_ms),
            "architecture": architecture,
            "training_config": training_config,
            "seed": seed,
            "data_contract": dict(checkpoint_metadata or {}),
        },
        checkpoint_path,
    )
    prediction = _predict(model, test, batch_size=batch_size, device=device)
    latency = _latency(model, test, device=device)
    total_model_pipeline_seconds = float(time.perf_counter() - started)
    details = {
        "device": device,
        "run_label": run_label,
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_sdp_backends": {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()) if torch.cuda.is_available() else None,
            "memory_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()) if torch.cuda.is_available() else None,
            "math": bool(torch.backends.cuda.math_sdp_enabled()) if torch.cuda.is_available() else None,
        },
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "loss": loss_config,
        "motion_reference": motion_reference,
        "epochs_completed": len(history),
        "best_val_loss": float(best_val),
        "training_seconds": optimization_seconds,
        "total_model_pipeline_seconds": total_model_pipeline_seconds,
        "latency_single_window": latency,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "progress_log": str(progress_path),
    }
    return prediction, details
