from __future__ import annotations

"""运行不依赖摄像头、Unity 或真机的一阶段离线验收。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from freihand.io import load_config, read_json, resolve_path, split_config_path, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单右手 baseline 的一阶段离线验收")
    parser.add_argument("--config", default="../configs/freihand_eval.yaml", help="FreiHAND 评测配置")
    parser.add_argument("--split", default="evaluation", help="验收 split，正式验收应为 evaluation")
    parser.add_argument("--max-samples", default=None, type=int, help="仅用于 smoke run 的样本上限")
    parser.add_argument("--output-root", default=None, help="独立验收运行目录的父目录")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 compileall 和 pytest，仅用于调试")
    parser.add_argument(
        "--require-full-split",
        action="store_true",
        help="要求全量 split，禁止 max-samples，并校验预测记录完整对齐",
    )
    return parser.parse_args()


def resolve_config_arg(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (THIS_DIR / path).resolve()


def resolve_cli_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    # porcelain 输出依靠行首的 XY 状态列；不能用 strip() 吃掉首行空格。
    return completed.stdout.rstrip("\r\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _listing_sha256(paths: list[Path], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        record = f"{path.relative_to(relative_to).as_posix()}\t{stat.st_size}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ["mediapipe", "opencv-contrib-python", "numpy", "Pillow", "PyYAML", "pytest"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _write_source_snapshot(run_dir: Path) -> dict[str, Any]:
    """记录本次实际使用的源码/配置内容，覆盖 dirty 和未跟踪文件。"""

    listed = _run_git(
        "-c",
        "core.quotepath=false",
        "ls-files",
        "-co",
        "--exclude-standard",
        "--",
        ".",
    ).splitlines()
    allowed_suffixes = {".py", ".yaml", ".yml", ".json", ".md", ".txt"}
    excluded_roots = {"dataset", "output", "outputs", "__pycache__"}
    records: list[tuple[str, str]] = []
    for value in sorted(set(listed)):
        relative = Path(value)
        if not relative.parts or relative.parts[0] in excluded_roots:
            continue
        if relative.suffix.lower() not in allowed_suffixes:
            continue
        path = PROJECT_ROOT / relative
        if path.is_file():
            records.append((relative.as_posix(), _sha256(path)))

    snapshot_path = run_dir / "source_checksums.sha256"
    write_text(
        snapshot_path,
        "".join(f"{digest}  {relative}\n" for relative, digest in records),
    )
    aggregate = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    return {
        "file_count": len(records),
        "aggregate_sha256": aggregate,
        "checksums_path": str(snapshot_path),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_run_id(git_commit: str, dirty: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    state = "dirty" if dirty else "clean"
    return f"{stamp}_{git_commit[:7]}_{state}"


def _create_run_dir(output_root: Path, run_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    return run_dir


def _run_command(name: str, command: list[str], *, run_dir: Path) -> dict[str, Any]:
    started_at = _utc_now()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    completed_at = _utc_now()
    log_path = run_dir / f"{name}.log"
    log_text = "\n".join(
        [
            "command:",
            json.dumps(command, ensure_ascii=False),
            "",
            "stdout:",
            completed.stdout.rstrip(),
            "",
            "stderr:",
            completed.stderr.rstrip(),
            "",
        ]
    )
    write_text(log_path, log_text)
    result = {
        "name": name,
        "command": command,
        "started_at": started_at,
        "completed_at": completed_at,
        "returncode": completed.returncode,
        "log": str(log_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"{name} 失败，returncode={completed.returncode}，详见 {log_path}")
    return result


def _threshold_value(mapping: dict[str, Any], threshold: float) -> float | None:
    for key, value in mapping.items():
        try:
            if abs(float(key) - float(threshold)) < 1.0e-9:
                return None if value is None else float(value)
        except (TypeError, ValueError):
            continue
    return None


def _render_acceptance_summary(
    metrics: dict[str, Any],
    checks: dict[str, bool],
    *,
    split: str,
    run_id: str,
    release_eligible: bool,
) -> str:
    counts = metrics.get("sample_counts", {})
    latency = metrics.get("latency_ms", {})
    pck_all = metrics.get("pck_2d_at_thresholds_all_gt", {})
    pck20_all = _threshold_value(pck_all, 20)
    gate_passed = all(checks.values())

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.3f}%"

    def num(value: Any, suffix: str = "") -> str:
        return "N/A" if value is None else f"{float(value):.3f}{suffix}"

    check_rows = [
        f"| {name} | {'通过' if passed else '未通过'} |"
        for name, passed in checks.items()
    ]
    return "\n".join(
        [
            "# 一阶段离线验收结果",
            "",
            f"- run_id：`{run_id}`",
            f"- split：`{split}`",
            f"- 阶段回归门槛：**{'通过' if gate_passed else '未通过'}**",
            f"- 可作为发布冻结结果：**{'是' if release_eligible else '否'}**",
            "",
            "## 核心结果",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| GT / prediction / matched | {counts.get('ground_truth_samples', 0)} / {counts.get('prediction_samples', 0)} / {counts.get('matched_prediction_samples', 0)} |",
            f"| 21 点完整率 | {pct(metrics.get('keypoint_complete_rate'))} |",
            f"| PCK@20px（全部 GT） | {pct(pck20_all)} |",
            f"| PCK@20px（仅有效预测） | {pct(_threshold_value(metrics.get('pck_2d_at_thresholds', {}), 20))} |",
            f"| 2D MPJPE（仅有效预测） | {num(metrics.get('mpjpe_2d_px'), ' px')} |",
            f"| 检测器加手选择 P95 | {num(latency.get('p95'), ' ms')} |",
            f"| 检测器加手选择 P99 | {num(latency.get('p99'), ' ms')} |",
            f"| 检测器加手选择最大值 | {num(latency.get('max'), ' ms')} |",
            "",
            "## 自动门槛",
            "",
            "| 检查项 | 结果 |",
            "| --- | --- |",
            *check_rows,
            "",
            "## 能证明什么",
            "",
            "- 当前代码、payload contract、单元/集成测试和本机 UDP loopback 可离线运行。",
            "- MediaPipe 在本次 FreiHAND 静态图像 split 上的 2D 关键点完整率、定位误差和检测阶段耗时。",
            "- 全部结果来自本 run 目录，不会覆盖其他实验。",
            "",
            "## 不能证明什么",
            "",
            "- 不证明实时视频稳定性、无手误检率、遮挡恢复或真实摄像头效果。",
            "- 不证明 Unity 渲染完成时间、实体 SVH/AUBO 响应或 5G 链路时延。",
            "- FreiHAND 使用静态单手图像且当前配置 prefer_any_hand=true；它不是运行期右手筛选策略验收。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.require_full_split and args.max_samples is not None:
        raise ValueError("--require-full-split 不能与 --max-samples 同时使用")

    config_path = resolve_config_arg(args.config)
    config = load_config(config_path)
    git_commit = _run_git("rev-parse", "HEAD")
    git_branch = _run_git("branch", "--show-current")
    git_status = _run_git("status", "--porcelain=v1")
    dirty = bool(git_status)
    run_id = _make_run_id(git_commit, dirty)
    output_root = (
        resolve_cli_path(args.output_root)
        if args.output_root
        else MODULE_ROOT / "outputs" / "phase1_acceptance"
    )
    run_dir = _create_run_dir(output_root, run_id)
    incomplete_marker = run_dir / ".incomplete"
    write_text(incomplete_marker, "本目录尚未完成验收；请查看 manifest.json。\n")

    image_root = split_config_path(config, args.split, "image_root")
    if image_root is None or not image_root.exists():
        raise FileNotFoundError(f"缺少 split={args.split} 的 image_root：{image_root}")
    extension = str(config.data.get("current_pipeline", {}).get("image_extension", ".jpg"))
    all_image_paths = sorted(image_root.glob(f"*{extension}"))
    selected_image_paths = (
        all_image_paths
        if args.max_samples is None
        else all_image_paths[: max(0, int(args.max_samples))]
    )

    effective_config_path = run_dir / "effective_config.yaml"
    write_text(
        effective_config_path,
        yaml.safe_dump(config.data, allow_unicode=True, sort_keys=False),
    )
    source_snapshot = _write_source_snapshot(run_dir)
    unity_receiver_path = Path(r"D:\SVH\RoboticArm\Assets\Scripts\RobotControlScript.cs")
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "scope": "single-right-hand phase1 offline acceptance",
        "release_eligible": False,
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "dirty": dirty,
            "status_porcelain": git_status.splitlines(),
        },
        "source_snapshot": source_snapshot,
        "unity_receiver_source": {
            "path": str(unity_receiver_path),
            "sha256": _sha256(unity_receiver_path) if unity_receiver_path.exists() else None,
        },
        "runtime": {
            "cwd": str(PROJECT_ROOT),
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "config": {
            "source_path": str(config_path),
            "source_sha256": _sha256(config_path),
            "effective_path": str(effective_config_path),
            "split": args.split,
            "max_samples": args.max_samples,
            "require_full_split": bool(args.require_full_split),
            "current_pipeline": config.data.get("current_pipeline", {}),
        },
        "dataset": {
            "image_root": str(image_root),
            "configured_extension": extension,
            "available_image_count": len(all_image_paths),
            "selected_image_count": len(selected_image_paths),
            "selected_listing_sha256": _listing_sha256(selected_image_paths, relative_to=image_root),
            "annotations": {},
        },
        "commands": [],
        "metrics_summary": None,
        "checks": None,
        "artifacts": {},
        "error": None,
    }
    for key in ["K", "xyz", "scale"]:
        annotation_path = split_config_path(config, args.split, key)
        manifest["dataset"]["annotations"][key] = {
            "path": None if annotation_path is None else str(annotation_path),
            "sha256": None
            if annotation_path is None or not annotation_path.exists()
            else _sha256(annotation_path),
        }
    write_json(manifest_path, manifest)

    predictions_path = run_dir / "predictions.json"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "eval_report.md"
    table_path = run_dir / "ppt_report_table.md"
    figures_dir = run_dir / "figures"
    python = sys.executable

    try:
        if not args.skip_tests:
            manifest["commands"].append(
                _run_command(
                    "compileall",
                    [python, "-m", "compileall", "-q", "src", "experiments/freihand_eval"],
                    run_dir=run_dir,
                )
            )
            manifest["commands"].append(
                _run_command("pytest", [python, "-m", "pytest", "-q"], run_dir=run_dir)
            )

        prediction_command = [
            python,
            str(THIS_DIR / "run_current_pipeline_predictions.py"),
            "--config",
            str(config_path),
            "--split",
            args.split,
            "--output",
            str(predictions_path),
        ]
        if args.max_samples is not None:
            prediction_command.extend(["--max-samples", str(args.max_samples)])
        manifest["commands"].append(
            _run_command("predict", prediction_command, run_dir=run_dir)
        )

        evaluation_command = [
            python,
            str(THIS_DIR / "evaluate_predictions.py"),
            "--config",
            str(config_path),
            "--split",
            args.split,
            "--predictions",
            str(predictions_path),
            "--metrics-output",
            str(metrics_path),
            "--report-output",
            str(report_path),
        ]
        if args.max_samples is not None:
            evaluation_command.extend(["--max-samples", str(args.max_samples)])
        if args.require_full_split:
            evaluation_command.append("--require-full-split")
        manifest["commands"].append(
            _run_command("evaluate", evaluation_command, run_dir=run_dir)
        )
        manifest["commands"].append(
            _run_command(
                "report_table",
                [
                    python,
                    str(THIS_DIR / "make_report_table.py"),
                    "--config",
                    str(config_path),
                    "--metrics",
                    str(metrics_path),
                    "--output",
                    str(table_path),
                ],
                run_dir=run_dir,
            )
        )
        manifest["commands"].append(
            _run_command(
                "report_figures",
                [
                    python,
                    str(THIS_DIR / "make_report_figures.py"),
                    "--config",
                    str(config_path),
                    "--metrics",
                    str(metrics_path),
                    "--output-dir",
                    str(figures_dir),
                ],
                run_dir=run_dir,
            )
        )

        metrics = read_json(metrics_path)
        counts = metrics.get("sample_counts", {})
        latency = metrics.get("latency_ms", {})
        pck20_all = _threshold_value(metrics.get("pck_2d_at_thresholds_all_gt", {}), 20)
        acceptance_cfg = config.data.get("phase1_acceptance", {})
        complete_rate = metrics.get("keypoint_complete_rate")
        mpjpe = metrics.get("mpjpe_2d_px")
        p95 = latency.get("p95")
        expected_samples = len(selected_image_paths)
        checks = {
            "源码编译检查": (
                not args.skip_tests
                and any(item["name"] == "compileall" and item["returncode"] == 0 for item in manifest["commands"])
            ),
            "pytest 全量测试": (
                not args.skip_tests
                and any(item["name"] == "pytest" and item["returncode"] == 0 for item in manifest["commands"])
            ),
            "预测与 GT 数量对齐": (
                int(counts.get("ground_truth_samples", -1)) == expected_samples
                and int(counts.get("prediction_samples", -1)) == expected_samples
                and int(counts.get("matched_prediction_samples", -1)) == expected_samples
                and int(counts.get("missing_prediction_samples", -1)) == 0
                and int(counts.get("extra_prediction_samples", -1)) == 0
            ),
            "每个样本都有耗时记录": int(latency.get("count", -1)) == expected_samples,
            "21 点完整率达到阶段下限": (
                complete_rate is not None
                and float(complete_rate) >= float(acceptance_cfg.get("minimum_complete_rate", 0.90))
            ),
            "全 GT PCK@20 达到阶段下限": (
                pck20_all is not None
                and pck20_all >= float(acceptance_cfg.get("minimum_pck20_all_gt", 0.80))
            ),
            "有效预测 MPJPE 不超过阶段上限": (
                mpjpe is not None
                and float(mpjpe) <= float(acceptance_cfg.get("maximum_mpjpe_2d_px", 10.0))
            ),
            "检测器加手选择 P95 不超过阶段上限": (
                p95 is not None
                and float(p95) <= float(acceptance_cfg.get("maximum_detection_p95_ms", 50.0))
            ),
        }
        full_split = args.max_samples is None and expected_samples == len(all_image_paths)
        release_eligible = bool(
            not dirty
            and full_split
            and args.split == "evaluation"
            and not args.skip_tests
            and all(checks.values())
        )
        manifest["checks"] = checks
        manifest["release_eligible"] = release_eligible
        manifest["metrics_summary"] = {
            "keypoint_complete_rate": complete_rate,
            "pck20_all_gt": pck20_all,
            "pck20_valid_predictions": _threshold_value(metrics.get("pck_2d_at_thresholds", {}), 20),
            "mpjpe_2d_px": mpjpe,
            "latency_ms": latency,
        }
        acceptance_path = run_dir / "acceptance.md"
        write_text(
            acceptance_path,
            _render_acceptance_summary(
                metrics,
                checks,
                split=args.split,
                run_id=run_id,
                release_eligible=release_eligible,
            ),
        )

        artifact_paths = [
            effective_config_path,
            Path(source_snapshot["checksums_path"]),
            predictions_path,
            metrics_path,
            report_path,
            table_path,
            acceptance_path,
            *sorted(figures_dir.glob("*.svg")),
            *sorted(run_dir.glob("*.log")),
        ]
        checksum_lines = []
        for artifact_path in artifact_paths:
            digest = _sha256(artifact_path)
            relative = artifact_path.relative_to(run_dir).as_posix()
            manifest["artifacts"][relative] = {
                "path": str(artifact_path),
                "sha256": digest,
                "bytes": artifact_path.stat().st_size,
            }
            checksum_lines.append(f"{digest}  {relative}")
        checksums_path = run_dir / "checksums.sha256"
        write_text(checksums_path, "\n".join(checksum_lines) + "\n")
        manifest["artifacts"]["checksums.sha256"] = {
            "path": str(checksums_path),
            "sha256": _sha256(checksums_path),
            "bytes": checksums_path.stat().st_size,
        }
        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        write_json(manifest_path, manifest)
        incomplete_marker.unlink()
        print(f"phase1 offline acceptance completed: {run_dir}")
        print(f"acceptance summary: {acceptance_path}")
        print(f"release eligible: {release_eligible}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = _utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
