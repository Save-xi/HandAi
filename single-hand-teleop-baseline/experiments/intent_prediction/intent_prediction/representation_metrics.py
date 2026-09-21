"""M2 共用 9 通道误差、逐 take 配对和研发继续条件。"""
from __future__ import annotations

import numpy as np


def control_metrics(batch, prediction: np.ndarray, prediction_valid: np.ndarray | None = None,
                    *, strata: dict[str, np.ndarray] | None = None) -> dict:
    pred = np.asarray(prediction, dtype=float)
    if pred.shape != batch.targets9.shape or not np.isfinite(pred).all():
        raise ValueError("共同通道预测形状不匹配或含非有限值")
    if np.any((pred < -1e-6) | (pred > 1 + 1e-6)):
        raise ValueError("9 通道预测须在 [0,1]；越界不能从分母排除")
    prediction_valid = np.ones(pred.shape[:2], dtype=bool) if prediction_valid is None else prediction_valid
    if prediction_valid.shape != pred.shape[:2]:
        raise ValueError("预测有效性与共同查询网格不匹配")
    per_horizon = {}
    for horizon in batch.points.eval_horizon_ms:
        index = int(np.flatnonzero(np.isclose(batch.query_ms, horizon, rtol=0, atol=1e-7))[0])
        mask = batch.target_valid[:, index]
        mse = np.mean((pred[:, index] - batch.targets9[:, index]) ** 2, axis=-1)
        per_sequence = {}
        for identity in np.unique(batch.points.sequence_ids[mask]):
            selected = mask & (batch.points.sequence_ids == identity)
            per_sequence[str(identity)] = float(np.sqrt(mse[selected].mean()))
        candidates = np.flatnonzero(mask)
        worst = candidates[np.argsort(mse[candidates])[-3:][::-1]] if len(candidates) else []
        per_horizon[str(horizon)] = {
            "sequence_equal_rmse": float(np.mean(list(per_sequence.values()))) if per_sequence else None,
            "frame_weighted_rmse": float(np.sqrt(mse[mask].mean())) if mask.any() else None,
            "per_frame_rmse_p95": float(np.percentile(np.sqrt(mse[mask]), 95)) if mask.any() else None,
            "per_sequence_rmse": per_sequence,
            "reference_valid_frames": int(mask.sum()), "eligible_input_windows": len(mask),
            "reference_coverage": float(mask.mean()),
            "prediction_valid_fraction": float(np.mean(prediction_valid[:, index])),
            "fallback_frames_with_reference": int(np.sum(mask & ~prediction_valid[:, index])),
            "worst_anchors": [{"sequence_id": str(batch.points.sequence_ids[n]),
                               "anchor_frame_id": int(batch.points.anchor_frame_ids[n]),
                               "rmse": float(np.sqrt(mse[n])),
                               "reference9": batch.targets9[n, index].tolist(),
                               "prediction9": pred[n, index].tolist()} for n in worst],
            "strata": {},
        }
        for name, selected in (strata or {}).items():
            selected = selected & mask
            errors = [float(np.sqrt(mse[selected & (batch.points.sequence_ids == identity)].mean()))
                      for identity in np.unique(batch.points.sequence_ids[selected])]
            per_horizon[str(horizon)]["strata"][name] = {
                "reference_valid_frames": int(selected.sum()), "sequence_count": len(errors),
                "sequence_equal_rmse": float(np.mean(errors)) if errors else None,
                "per_frame_rmse_p95": float(np.percentile(np.sqrt(mse[selected]), 95)) if errors else None}
    primary = [per_horizon[str(h)]["sequence_equal_rmse"] for h in (50, 100)]
    if any(value is None for value in primary):
        raise ValueError("50/100ms 没有可比较的共同参考")
    return {"selection_score": float(np.mean(primary)), "per_horizon_ms": per_horizon,
            "denominator": "same_reference_valid_anchors_for_all_routes_including_prediction_fallbacks"}


def continuation_summary(baseline: dict, candidates: list[dict], *, seed: int, minimum_seeds: int = 3) -> dict:
    """按原始 take 分组；训练种子只先求均值，不伪装成新增人员。"""
    base_score = baseline["selection_score"]
    scores = np.asarray([candidate["selection_score"] for candidate in candidates])
    improvement = None if base_score <= 1e-8 else float(1 - scores.mean() / base_score)
    individual_horizons, tail = {}, {}
    for h in (50, 100):
        row = baseline["per_horizon_ms"][str(h)]
        values = [candidate["per_horizon_ms"][str(h)] for candidate in candidates]
        individual_horizons[str(h)] = bool(np.mean([v["sequence_equal_rmse"] for v in values]) <= row["sequence_equal_rmse"] + 1e-8)
        tail[str(h)] = bool(np.mean([v["per_frame_rmse_p95"] for v in values]) <= row["per_frame_rmse_p95"] * 1.02 + 1e-8)
    identities = sorted(set(baseline["per_horizon_ms"]["50"]["per_sequence_rmse"]) &
                        set(baseline["per_horizon_ms"]["100"]["per_sequence_rmse"]))
    baseline_seq, candidate_seq, groups = [], [], []
    for identity in identities:
        if not all(identity in candidate["per_horizon_ms"][str(h)]["per_sequence_rmse"]
                   for candidate in candidates for h in (50, 100)):
            raise ValueError("配对比较的序列分母发生变化")
        baseline_seq.append(np.mean([baseline["per_horizon_ms"][str(h)]["per_sequence_rmse"][identity] for h in (50, 100)]))
        candidate_seq.append(np.mean([candidate["per_horizon_ms"][str(h)]["per_sequence_rmse"][identity]
                                     for candidate in candidates for h in (50, 100)]))
        groups.append(identity.rsplit("_segment", 1)[0])
    baseline_seq, candidate_seq, groups = np.asarray(baseline_seq), np.asarray(candidate_seq), np.asarray(groups)
    unique = np.unique(groups)
    deltas = baseline_seq - candidate_seq
    wins = sum(float(deltas[groups == group].mean()) > 0 for group in unique)
    ci = None
    if len(unique) >= 2:
        rng = np.random.default_rng(seed)
        sums = np.array([deltas[groups == group].sum() for group in unique])
        counts = np.array([(groups == group).sum() for group in unique])
        draws = rng.integers(0, len(unique), (2000, len(unique)))
        samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        ci = np.quantile(samples, [0.025, 0.975]).tolist()
    checks = {"at_least_three_training_seeds": len(candidates) >= minimum_seeds,
              "mean_improvement_at_least_5_percent": improvement is not None and improvement >= 0.05,
              "both_horizons_no_regression": all(individual_horizons.values()),
              "both_tail_regressions_at_most_2_percent": all(tail.values()),
              "majority_takes_improve": wins > len(unique) / 2,
              "majority_seeds_improve": int(np.sum(scores < base_score)) > len(scores) / 2}
    return {"seed_count": len(scores), "mean_score": float(scores.mean()),
            "seed_std": float(scores.std(ddof=1)) if len(scores) > 1 else None,
            "improvement_fraction": improvement, "absolute_improvement": float(base_score - scores.mean()),
            "paired_take_count": len(unique), "improving_takes": int(wins),
            "paired_take_bootstrap_95ci_absolute_improvement": ci,
            "bootstrap_scope": "conditional_on_recorded_subjects_and_mean_trained_models_not_cross_person_generalization",
            "checks": checks, "development_continue_criteria_passed": all(checks.values())}
