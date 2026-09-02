from __future__ import annotations

"""真实摄像头域盲测的脱离 Git 树冻结清单与一次性运行凭据。

冻结清单必须在验证代码、协议和机器配置合并后生成，因此它放在 Git 跟踪树外，
并由调用者显式提供 SHA-256。这样既能记录最终 Git revision，又不会产生“文件中
必须预先写入包含它自己的 commit SHA”这一自引用问题。
"""

from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping
import uuid

from prediction.shadow_predictor import build_prediction_shadow
from svh.mapping_contract import (
    MAPPING_CONTRACT_VERSION,
    assert_mapping_implementation_compatible,
    mapping_contract_sha256,
)
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLIND_FREEZE_SCHEMA_VERSION = "handai-camera-domain-blind-freeze-v1"
BLIND_ATTEMPT_RECEIPT_SCHEMA_VERSION = "handai-camera-domain-blind-attempt-v1"
BLIND_FREEZE_STATE_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "intent_prediction"
    / "outputs"
    / "camera_domain_blind_freeze"
)

_RUNTIME_PATH_KEYS = {
    "video_file_path",
    "output_json_path",
    "jsonl_output_dir",
    "prediction_shadow_output_json_path",
    "prediction_shadow_jsonl_output_dir",
    "prediction_shadow_selection_path",
    "prediction_shadow_checkpoint_path",
    "prediction_shadow_report_path",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象：{path}")
    return value


def resolve_relative(base_file: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_file.resolve().parent / candidate).resolve()


def prepare_effective_runtime_config(runtime_config_path: Path) -> dict[str, Any]:
    """复现 camera-domain 评测固定使用的旁路运行配置。"""

    cfg = load_config(str(runtime_config_path.resolve()))
    cfg["unity_udp_enabled"] = False
    cfg["save_last_json"] = False
    cfg["save_jsonl"] = False
    cfg["prediction_shadow_enabled"] = True
    cfg["enable_control_extension"] = True
    cfg["svh_enable_preview"] = True
    return cfg


def _manifest_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        stored_path = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        scope = "project_relative"
    except ValueError:
        stored_path = str(resolved)
        scope = "absolute"
    return {
        "path": stored_path,
        "path_scope": scope,
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _resolve_manifest_path(record: Mapping[str, Any]) -> Path:
    raw_path = record.get("path")
    scope = record.get("path_scope")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("freeze manifest artifact 缺少 path")
    if scope == "project_relative":
        return (PROJECT_ROOT / Path(raw_path)).resolve()
    if scope == "absolute":
        return Path(raw_path).resolve()
    raise ValueError(f"freeze manifest artifact path_scope 不受支持：{scope}")


def _artifact_record(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"冻结工件不存在（{label}）：{resolved}")
    return _manifest_path(resolved)


def _normalize_runtime_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(cfg)
    for key in _RUNTIME_PATH_KEYS:
        value = normalized.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).resolve()
        try:
            normalized[key] = {
                "project_relative": path.relative_to(PROJECT_ROOT.resolve()).as_posix()
            }
        except ValueError:
            normalized[key] = {"absolute": str(path)}
    return normalized


def effective_runtime_config_sha256(cfg: Mapping[str, Any]) -> str:
    return json_sha256(_normalize_runtime_config(cfg))


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _canonical_command_output(value: str) -> str:
    lines = [line.rstrip() for line in value.splitlines()]
    return "\n".join(lines).strip() + "\n"


def _conda_executable() -> Path | None:
    candidates: list[Path] = []
    for raw in (os.environ.get("CONDA_EXE"), os.environ.get("_CONDA_EXE")):
        if raw:
            candidates.append(Path(raw))
    prefix = Path(sys.prefix).resolve()
    if prefix.parent.name.casefold() == "envs":
        conda_root = prefix.parent.parent
        candidates.extend(
            [
                conda_root / "Scripts" / "conda.exe",
                conda_root / "condabin" / "conda.bat",
                conda_root / "bin" / "conda",
            ]
        )
    discovered = shutil.which("conda")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _command_fingerprint(command: list[str]) -> tuple[str | None, int, str | None]:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, 0, f"{type(exc).__name__}: {exc}"
    normalized = _canonical_command_output(completed.stdout)
    if not normalized.strip():
        return None, 0, "command returned no lock content"
    return (
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        len(normalized.splitlines()),
        None,
    )


def environment_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    conda_executable = _conda_executable()
    conda_sha256: str | None = None
    conda_line_count = 0
    conda_error: str | None = "conda executable not found"
    if conda_executable is not None:
        conda_sha256, conda_line_count, conda_error = _command_fingerprint(
            [
                str(conda_executable),
                "list",
                "--explicit",
                "--prefix",
                str(prefix),
            ]
        )
    pip_sha256, pip_line_count, pip_error = _command_fingerprint(
        [str(executable), "-m", "pip", "freeze", "--all"]
    )
    identity: dict[str, str | int | bool | None] = {
        "python": platform.python_version(),
        "python_executable": str(executable),
        "python_executable_sha256": file_sha256(executable),
        "python_prefix": str(prefix),
        "conda_prefix": str(Path(os.environ.get("CONDA_PREFIX", str(prefix))).resolve()),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV") or prefix.name,
        "conda_environment_name": prefix.name,
        "conda_executable": (
            str(conda_executable) if conda_executable is not None else None
        ),
        "conda_explicit_spec_sha256": conda_sha256,
        "conda_explicit_spec_line_count": conda_line_count,
        "conda_explicit_spec_error": conda_error,
        "pip_freeze_sha256": pip_sha256,
        "pip_freeze_line_count": pip_line_count,
        "pip_freeze_error": pip_error,
        "numpy": _package_version("numpy"),
        "opencv_contrib_python": _package_version("opencv-contrib-python"),
        "mediapipe": _package_version("mediapipe"),
        "torch": _package_version("torch"),
    }
    try:
        import cv2

        identity["opencv_runtime"] = cv2.__version__
    except (ImportError, AttributeError):
        identity["opencv_runtime"] = None
    try:
        import torch

        identity["torch_cuda_version"] = torch.version.cuda
        identity["cuda_available"] = bool(torch.cuda.is_available())
        identity["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except (ImportError, RuntimeError):
        identity["torch_cuda_version"] = None
        identity["cuda_available"] = False
        identity["cuda_device_name"] = None
    return identity


def _validate_required_environment(
    environment: Mapping[str, Any],
    *,
    required_conda_environment: str,
) -> None:
    actual_name = environment.get("conda_environment_name")
    if actual_name != required_conda_environment:
        raise RuntimeError(
            "正式盲测必须使用 Conda 环境 "
            f"{required_conda_environment}，当前为 {actual_name!r}"
        )
    for field in (
        "python_executable_sha256",
        "conda_explicit_spec_sha256",
        "pip_freeze_sha256",
    ):
        value = environment.get(field)
        if not isinstance(value, str) or not re_full_sha256(value.lower()):
            raise RuntimeError(f"正式盲测环境未形成可验证锁：{field}")


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        tree = run("rev-parse", "HEAD^{tree}")
        origin_main = run("rev-parse", "--verify", "refs/remotes/origin/main")
        branch = run("branch", "--show-current")
        tracked_status = run("status", "--short", "--untracked-files=no")
        all_status = run("status", "--short")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "revision": revision,
        "tree": tree,
        "origin_main": origin_main,
        "branch": branch,
        "tracked_worktree_clean": not bool(tracked_status),
        "worktree_clean": not bool(all_status),
        "tracked_status_lines": tracked_status.splitlines() if tracked_status else [],
        "untracked_status_lines": [
            line for line in all_status.splitlines() if line.startswith("??")
        ],
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _freeze_artifact_paths(
    *,
    evaluation_config_path: Path,
    protocol_path: Path,
    runtime_config_path: Path,
    delay_config_path: Path,
    runtime_cfg: Mapping[str, Any],
) -> dict[str, Path]:
    return {
        "evaluation_config": evaluation_config_path.resolve(),
        "protocol": protocol_path.resolve(),
        "runtime_config": runtime_config_path.resolve(),
        "delay_config": delay_config_path.resolve(),
        "selection": Path(str(runtime_cfg["prediction_shadow_selection_path"])).resolve(),
        "checkpoint": Path(str(runtime_cfg["prediction_shadow_checkpoint_path"])).resolve(),
        "second_round_report": Path(
            str(runtime_cfg["prediction_shadow_report_path"])
        ).resolve(),
    }


def _blind_video_records(
    *,
    expected_ids: list[str],
    blind_video_paths: Mapping[str, Path],
    forbidden_sha256: set[str],
) -> dict[str, dict[str, Any]]:
    if set(blind_video_paths) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(blind_video_paths))
        unexpected = sorted(set(blind_video_paths) - set(expected_ids))
        raise ValueError(
            f"blind freeze 必须恰好封存 B1-B7；missing={missing}, unexpected={unexpected}"
        )
    records: dict[str, dict[str, Any]] = {}
    seen_sha256: dict[str, str] = {}
    for video_id in expected_ids:
        record = _artifact_record(blind_video_paths[video_id], label=video_id)
        digest = str(record["sha256"]).lower()
        if digest in forbidden_sha256:
            raise ValueError(f"{video_id} 与开发集 V1-V7 原始视频 SHA 重合")
        duplicate = seen_sha256.get(digest)
        if duplicate is not None:
            raise ValueError(f"{video_id} 与 {duplicate} 视频内容 SHA 重复")
        seen_sha256[digest] = video_id
        records[video_id] = record
    return records


def _content_identity_records(records: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for item_id in sorted(records):
        record = records[item_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"{label}.{item_id} 不是对象")
        digest = str(record.get("sha256", "")).lower()
        byte_count = record.get("bytes")
        if not re_full_sha256(digest) or not isinstance(byte_count, int):
            raise ValueError(f"{label}.{item_id} 缺少 SHA-256/bytes 身份")
        normalized[str(item_id)] = {"sha256": digest, "bytes": byte_count}
    return normalized


def attempt_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """提取与路径、时间和 manifest 文件名无关的正式尝试身份。"""

    git = manifest.get("git")
    artifacts = manifest.get("artifacts")
    blind_inputs = manifest.get("blind_inputs")
    videos = blind_inputs.get("videos") if isinstance(blind_inputs, Mapping) else None
    if not isinstance(git, Mapping):
        raise ValueError("blind freeze manifest 缺少 Git 身份")
    if not isinstance(artifacts, Mapping):
        raise ValueError("blind freeze manifest 缺少 artifacts 身份")
    if not isinstance(videos, Mapping):
        raise ValueError("blind freeze manifest 缺少视频身份")
    return {
        "schema_version": manifest.get("schema_version"),
        "claim_status": manifest.get("claim_status"),
        "git": {
            "revision": git.get("revision"),
            "tree": git.get("tree"),
            "origin_main_at_freeze": git.get("origin_main_at_freeze"),
        },
        "artifacts": _content_identity_records(artifacts, label="artifacts"),
        "evaluation": manifest.get("evaluation"),
        "blind_videos": _content_identity_records(videos, label="blind_videos"),
        "model_identity": manifest.get("model_identity"),
        "environment": manifest.get("environment"),
        "safety": manifest.get("safety"),
    }


def attempt_identity_sha256(manifest: Mapping[str, Any]) -> str:
    return json_sha256(attempt_identity_payload(manifest))


def default_blind_freeze_manifest_path(attempt_token: str) -> Path:
    token = str(attempt_token).lower()
    if not re_full_sha256(token):
        raise ValueError("attempt_token 必须是 64 位十六进制身份")
    return (BLIND_FREEZE_STATE_ROOT / "manifests" / f"{token}.json").resolve()


def default_attempt_receipt_path(attempt_token: str) -> Path:
    token = str(attempt_token).lower()
    if not re_full_sha256(token):
        raise ValueError("attempt_token 必须是 64 位十六进制身份")
    return (BLIND_FREEZE_STATE_ROOT / "attempt_receipts" / f"{token}.json").resolve()


def _set_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _remove_partial_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    path.unlink(missing_ok=True)


def stage_blind_video_inputs(
    blind_video_paths: Mapping[str, Path],
    freeze_state: Mapping[str, Any],
) -> dict[str, Path]:
    """复制并核验只读内容寻址副本，正式解码永不再打开原始 B 路径。"""

    attempt_token = str(freeze_state.get("attempt_token", "")).lower()
    if not re_full_sha256(attempt_token):
        raise ValueError("freeze state 缺少确定性 attempt_token")
    blind_inputs = freeze_state.get("blind_inputs")
    frozen_videos = (
        blind_inputs.get("videos") if isinstance(blind_inputs, Mapping) else None
    )
    if not isinstance(frozen_videos, Mapping):
        raise ValueError("freeze state 缺少 B1-B7 输入身份")
    if set(blind_video_paths) != set(frozen_videos):
        raise ValueError("待封存的 B1-B7 输入集合与 freeze state 不一致")

    sealed_root = (BLIND_FREEZE_STATE_ROOT / "sealed_inputs" / attempt_token).resolve()
    sealed_root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for video_id in sorted(frozen_videos):
        record = frozen_videos[video_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"freeze state 的 {video_id} 输入记录非法")
        source = blind_video_paths[str(video_id)].resolve()
        _verify_artifact(str(video_id), record, expected_path=source)
        digest = str(record.get("sha256", "")).lower()
        byte_count = int(record.get("bytes", -1))
        safe_id = str(video_id).replace("/", "_").replace("\\", "_")
        suffix = source.suffix.lower() or ".bin"
        destination = sealed_root / f"{safe_id}-{digest}{suffix}"
        created = False
        try:
            with source.open("rb") as source_handle, destination.open("xb") as output:
                created = True
                shutil.copyfileobj(source_handle, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        except FileExistsError:
            pass
        except BaseException:
            if created:
                _remove_partial_file(destination)
            raise
        try:
            if destination.stat().st_size != byte_count:
                raise ValueError(f"{video_id} 内容寻址副本字节数与 freeze manifest 不一致")
            if file_sha256(destination) != digest:
                raise ValueError(f"{video_id} 内容寻址副本 SHA-256 与 freeze manifest 不一致")
            _set_read_only(destination)
        except BaseException:
            if created:
                _remove_partial_file(destination)
            raise
        staged[str(video_id)] = destination.resolve()
    return staged


def create_blind_freeze_manifest(
    *,
    evaluation_config_path: Path,
    protocol_path: Path,
    runtime_config_path: Path,
    delay_config_path: Path,
    blind_video_paths: Mapping[str, Path],
    output_path: Path | None = None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """在已合并且 tracked-clean 的目标 revision 上生成脱离 Git 树的冻结清单。"""

    evaluation_config_path = evaluation_config_path.resolve()
    evaluation_config = read_json_object(evaluation_config_path)
    blind_policy = evaluation_config.get("blind_policy")
    if evaluation_config.get("protocol_stage") != "blind_frozen":
        raise ValueError("只有 protocol_stage=blind_frozen 的配置才能生成盲测冻结清单")
    if not isinstance(blind_policy, dict) or blind_policy.get("enabled") is not True:
        raise ValueError("blind_policy 尚未启用")
    if not isinstance(blind_policy.get("gate"), dict):
        raise ValueError("blind_policy.gate 尚未冻结")
    required_conda_environment = evaluation_config.get("required_conda_environment")
    if not isinstance(required_conda_environment, str) or not required_conda_environment:
        raise ValueError("正式盲测配置缺少 required_conda_environment")
    if not isinstance(evaluation_config.get("input_mirrored"), bool):
        raise ValueError("正式盲测配置必须冻结 input_mirrored")

    git = git_snapshot()
    if git.get("available") is not True:
        raise RuntimeError("无法读取 Git 身份，拒绝生成盲测冻结清单")
    if git.get("worktree_clean") is not True:
        raise RuntimeError("Git 工作树存在 tracked 或非忽略 untracked 文件，拒绝生成盲测冻结清单")
    if git.get("revision") != git.get("origin_main"):
        raise RuntimeError("只能在已获取且检出的 origin/main 最终合并 revision 上生成冻结清单")

    environment = environment_identity()
    _validate_required_environment(
        environment,
        required_conda_environment=required_conda_environment,
    )
    runtime_cfg = prepare_effective_runtime_config(runtime_config_path)
    if bool(runtime_cfg.get("input_mirrored", False)) != evaluation_config[
        "input_mirrored"
    ]:
        raise ValueError("runtime config 的 input_mirrored 与盲测配置不一致")
    predictor = build_prediction_shadow(runtime_cfg, logger=logger)
    if predictor is None or predictor.initialization_error is not None:
        detail = None if predictor is None else predictor.initialization_error
        raise RuntimeError(f"无法加载冻结 prediction shadow：{detail}")
    second_round_report_path = Path(
        str(runtime_cfg["prediction_shadow_report_path"])
    ).resolve()
    second_round_report = read_json_object(second_round_report_path)
    dynamic_q90 = second_round_report.get("motion_strata_thresholds_from_validation", {}).get(
        "q90"
    )
    if not isinstance(dynamic_q90, (int, float)) or not math.isfinite(float(dynamic_q90)):
        raise ValueError("second-round report 缺少有限 validation q90")

    mapping_implementation_sha256 = assert_mapping_implementation_compatible(runtime_cfg)
    artifact_paths = _freeze_artifact_paths(
        evaluation_config_path=evaluation_config_path,
        protocol_path=protocol_path,
        runtime_config_path=runtime_config_path,
        delay_config_path=delay_config_path,
        runtime_cfg=runtime_cfg,
    )
    artifacts = {
        label: _artifact_record(path, label=label)
        for label, path in artifact_paths.items()
    }
    if predictor.selection_sha256 != artifacts["selection"]["sha256"]:
        raise ValueError("predictor selection SHA 与实际文件不一致")
    if predictor.checkpoint_sha256 != artifacts["checkpoint"]["sha256"]:
        raise ValueError("predictor checkpoint SHA 与实际文件不一致")

    video_ids = evaluation_config.get("video_sets", {}).get("blind")
    if not isinstance(video_ids, list) or not video_ids:
        raise ValueError("evaluation config 缺少 blind video set")
    expected_video_ids = [str(value) for value in video_ids]
    forbidden_sha256 = {
        str(value).lower()
        for value in blind_policy.get("forbidden_video_sha256", [])
    }
    blind_videos = _blind_video_records(
        expected_ids=expected_video_ids,
        blind_video_paths=blind_video_paths,
        forbidden_sha256=forbidden_sha256,
    )
    created_at_utc = utc_now_iso()
    evaluation_identity = {
        "config_schema_version": evaluation_config.get("schema_version"),
        "protocol_stage": evaluation_config.get("protocol_stage"),
        "blind_video_ids": expected_video_ids,
        "blind_policy_sha256": json_sha256(blind_policy),
        "input_mirrored": bool(runtime_cfg.get("input_mirrored", False)),
        "required_conda_environment": required_conda_environment,
        "effective_runtime_config_sha256": effective_runtime_config_sha256(runtime_cfg),
    }
    model_identity = {
        "model_label": predictor.model_label,
        "device": predictor.device,
        "history_frames": int(predictor.history_frames),
        "horizon_ms": [int(value) for value in predictor.horizon_ms],
        "selection_sha256": predictor.selection_sha256,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "second_round_report_sha256": artifacts["second_round_report"]["sha256"],
        "validation_q90": float(dynamic_q90),
        "mapping_contract_version": MAPPING_CONTRACT_VERSION,
        "mapping_contract_sha256": mapping_contract_sha256(runtime_cfg),
        "mapping_implementation_sha256": mapping_implementation_sha256,
        "offline_gate_passed": bool(
            second_round_report.get("acceptance", {}).get("offline_gate_passed", False)
        ),
    }
    safety = {
        "single_right_hand_only": True,
        "prediction_shadow_only": True,
        "udp_created_by_evaluator": False,
        "prediction_modifies_unity_udp": False,
        "real_svh_in_scope": False,
    }
    payload: dict[str, Any] = {
        "schema_version": BLIND_FREEZE_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "claim_status": "camera_domain_pseudo_ground_truth_shadow_only",
        "git": {
            "revision": git["revision"],
            "tree": git["tree"],
            "origin_main_at_freeze": git["origin_main"],
            "branch_at_freeze": git["branch"],
            "require_worktree_clean": True,
            "untracked_status_lines_at_freeze": git["untracked_status_lines"],
        },
        "artifacts": artifacts,
        "evaluation": evaluation_identity,
        "blind_inputs": {
            "sealed_at_utc": created_at_utc,
            "videos": blind_videos,
        },
        "model_identity": model_identity,
        "environment": environment,
        "safety": safety,
    }
    identity_sha256 = attempt_identity_sha256(payload)
    payload["attempt_identity_sha256"] = identity_sha256
    payload["attempt_token"] = identity_sha256
    canonical_output_path = default_blind_freeze_manifest_path(identity_sha256)
    if output_path is not None and output_path.resolve() != canonical_output_path:
        raise ValueError(
            "blind freeze manifest 路径不可覆盖；固定路径为 "
            f"{canonical_output_path}"
        )
    try:
        _atomic_create_json(canonical_output_path, payload)
    except FileExistsError as exc:
        raise RuntimeError(f"冻结清单身份已登记，拒绝重复生成：{canonical_output_path}") from exc
    return {
        "path": str(canonical_output_path),
        "sha256": file_sha256(canonical_output_path),
        "attempt_receipt_path": str(default_attempt_receipt_path(identity_sha256)),
        "manifest": payload,
    }


def _verify_artifact(
    label: str,
    record: Mapping[str, Any],
    *,
    expected_path: Path,
) -> None:
    path = _resolve_manifest_path(record)
    expected = expected_path.resolve()
    if path != expected:
        raise ValueError(f"freeze manifest {label} 路径漂移：{path} != {expected}")
    if not path.is_file():
        raise FileNotFoundError(f"freeze manifest {label} 文件不存在：{path}")
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or file_sha256(path) != expected_sha.lower():
        raise ValueError(f"freeze manifest {label} SHA-256 漂移")
    expected_bytes = record.get("bytes")
    if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
        raise ValueError(f"freeze manifest {label} 字节数漂移")


def verify_blind_freeze_manifest(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    evaluation_config_path: Path,
    protocol_path: Path,
    runtime_config_path: Path,
    delay_config_path: Path,
    evaluation_config: Mapping[str, Any],
    input_mirrored: bool,
    blind_video_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """在任何 B 视频解码和输出目录创建之前 fail-closed 核对冻结身份。"""

    manifest_path = manifest_path.resolve()
    expected_sha = str(expected_manifest_sha256).strip().lower()
    if not re_full_sha256(expected_sha):
        raise ValueError("--expected-freeze-manifest-sha256 必须是 64 位十六进制")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"blind freeze manifest 不存在：{manifest_path}")
    actual_sha = file_sha256(manifest_path)
    if actual_sha != expected_sha:
        raise ValueError("blind freeze manifest SHA-256 与预期不一致")
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != BLIND_FREEZE_SCHEMA_VERSION:
        raise ValueError("blind freeze manifest schema_version 不受支持")
    if manifest.get("claim_status") != "camera_domain_pseudo_ground_truth_shadow_only":
        raise ValueError("blind freeze manifest claim_status 不受支持")
    attempt_token = str(manifest.get("attempt_token", "")).lower()
    frozen_identity = str(manifest.get("attempt_identity_sha256", "")).lower()
    computed_identity = attempt_identity_sha256(manifest)
    if not re_full_sha256(attempt_token) or attempt_token != computed_identity:
        raise ValueError("blind freeze manifest 的确定性 attempt_token 身份不一致")
    if frozen_identity != computed_identity:
        raise ValueError("blind freeze manifest 的 attempt_identity_sha256 不一致")
    canonical_manifest_path = default_blind_freeze_manifest_path(attempt_token)
    if manifest_path != canonical_manifest_path:
        raise ValueError(
            "blind freeze manifest 必须使用确定性固定路径："
            f"{canonical_manifest_path}"
        )

    current_git = git_snapshot()
    frozen_git = manifest.get("git")
    if current_git.get("available") is not True or not isinstance(frozen_git, dict):
        raise ValueError("blind freeze manifest 缺少可验证 Git 身份")
    if current_git.get("worktree_clean") is not True:
        raise ValueError("Git 工作树存在 tracked 或非忽略 untracked 文件，拒绝正式盲测")
    for key in ("revision", "tree"):
        if current_git.get(key) != frozen_git.get(key):
            raise ValueError(f"Git {key} 与 blind freeze manifest 不一致")
    if frozen_git.get("origin_main_at_freeze") != frozen_git.get("revision"):
        raise ValueError("blind freeze manifest 不是在 origin/main 最终合并 revision 上生成")
    if frozen_git.get("require_worktree_clean") is not True:
        raise ValueError("blind freeze manifest 未要求完整工作树 clean")

    runtime_cfg = prepare_effective_runtime_config(runtime_config_path)
    artifact_paths = _freeze_artifact_paths(
        evaluation_config_path=evaluation_config_path,
        protocol_path=protocol_path,
        runtime_config_path=runtime_config_path,
        delay_config_path=delay_config_path,
        runtime_cfg=runtime_cfg,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("blind freeze manifest 缺少 artifacts")
    for label, expected_path in artifact_paths.items():
        record = artifacts.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"blind freeze manifest 缺少 artifact：{label}")
        _verify_artifact(label, record, expected_path=expected_path)

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("blind freeze manifest 缺少 evaluation 身份")
    if evaluation.get("protocol_stage") != "blind_frozen":
        raise ValueError("blind freeze manifest 不是 blind_frozen")
    expected_ids = [str(value) for value in evaluation_config.get("video_sets", {}).get("blind", [])]
    if evaluation.get("blind_video_ids") != expected_ids:
        raise ValueError("blind video IDs 与 freeze manifest 不一致")
    if evaluation.get("blind_policy_sha256") != json_sha256(
        evaluation_config.get("blind_policy")
    ):
        raise ValueError("blind policy 与 freeze manifest 不一致")
    required_conda_environment = evaluation_config.get("required_conda_environment")
    if (
        not isinstance(required_conda_environment, str)
        or not required_conda_environment
        or evaluation.get("required_conda_environment")
        != required_conda_environment
    ):
        raise ValueError("required_conda_environment 与 freeze manifest 不一致")
    if not isinstance(evaluation_config.get("input_mirrored"), bool) or bool(
        evaluation_config["input_mirrored"]
    ) != bool(evaluation.get("input_mirrored")):
        raise ValueError("盲测配置的 input_mirrored 与 freeze manifest 不一致")
    if bool(evaluation.get("input_mirrored")) != bool(input_mirrored):
        raise ValueError("input_mirrored 与 freeze manifest 不一致")
    if evaluation.get("effective_runtime_config_sha256") != effective_runtime_config_sha256(
        runtime_cfg
    ):
        raise ValueError("effective runtime config 与 freeze manifest 不一致")
    current_environment = environment_identity()
    _validate_required_environment(
        current_environment,
        required_conda_environment=required_conda_environment,
    )
    if manifest.get("environment") != current_environment:
        raise ValueError("Python/NumPy/OpenCV/MediaPipe/PyTorch 环境与 freeze manifest 不一致")

    blind_inputs = manifest.get("blind_inputs")
    if not isinstance(blind_inputs, dict) or not isinstance(
        blind_inputs.get("videos"), dict
    ):
        raise ValueError("blind freeze manifest 缺少封存 B1-B7 输入")
    frozen_videos = blind_inputs["videos"]
    if set(frozen_videos) != set(expected_ids):
        raise ValueError("blind freeze manifest 的 B1-B7 输入集合不完整")
    if set(blind_video_paths) != set(expected_ids):
        raise ValueError("正式盲测提供的 B1-B7 输入集合不完整")
    seen_video_sha256: set[str] = set()
    forbidden_sha256 = {
        str(value).lower()
        for value in evaluation_config.get("blind_policy", {}).get(
            "forbidden_video_sha256", []
        )
    }
    for video_id in expected_ids:
        record = frozen_videos.get(video_id)
        if not isinstance(record, dict):
            raise ValueError(f"blind freeze manifest 缺少输入：{video_id}")
        path = blind_video_paths.get(video_id)
        if path is None:
            raise ValueError(f"正式盲测未提供输入：{video_id}")
        _verify_artifact(video_id, record, expected_path=path)
        digest = str(record.get("sha256", "")).lower()
        if digest in forbidden_sha256:
            raise ValueError(f"{video_id} 与开发集 V1-V7 原始视频 SHA 重合")
        if digest in seen_video_sha256:
            raise ValueError("blind freeze manifest 内存在重复 B 视频 SHA")
        seen_video_sha256.add(digest)

    model_identity = manifest.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError("blind freeze manifest 缺少 model_identity")
    if model_identity.get("mapping_contract_version") != MAPPING_CONTRACT_VERSION:
        raise ValueError("mapping contract version 与 freeze manifest 不一致")
    if model_identity.get("mapping_contract_sha256") != mapping_contract_sha256(runtime_cfg):
        raise ValueError("mapping contract SHA 与 freeze manifest 不一致")
    current_mapping_impl = assert_mapping_implementation_compatible(runtime_cfg)
    if model_identity.get("mapping_implementation_sha256") != current_mapping_impl:
        raise ValueError("mapping implementation SHA 与 freeze manifest 不一致")

    safety = manifest.get("safety")
    required_safety = {
        "single_right_hand_only": True,
        "prediction_shadow_only": True,
        "udp_created_by_evaluator": False,
        "prediction_modifies_unity_udp": False,
        "real_svh_in_scope": False,
    }
    if safety != required_safety:
        raise ValueError("blind freeze manifest 安全边界不完整")
    return {
        "path": str(manifest_path),
        "sha256": actual_sha,
        "attempt_token": attempt_token,
        "attempt_identity_sha256": computed_identity,
        "attempt_receipt_path": str(default_attempt_receipt_path(attempt_token)),
        "git": current_git,
        "artifacts": artifacts,
        "model_identity": model_identity,
        "environment": current_environment,
        "blind_inputs": blind_inputs,
    }


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def verify_algorithm_identity(
    algorithm: Mapping[str, Any],
    freeze_state: Mapping[str, Any],
) -> None:
    frozen = freeze_state.get("model_identity")
    actual = algorithm.get("model_identity")
    if not isinstance(frozen, Mapping) or not isinstance(actual, Mapping):
        raise ValueError("算法结果缺少可核对的 model_identity")
    field_pairs = {
        "model_label": "model_label",
        "device": "device",
        "history_frames": "history_frames",
        "horizon_ms": "horizon_ms",
        "selection_sha256": "selection_sha256",
        "checkpoint_sha256": "checkpoint_sha256",
        "offline_gate_passed": "offline_gate_passed",
    }
    for actual_key, frozen_key in field_pairs.items():
        if actual.get(actual_key) != frozen.get(frozen_key):
            raise ValueError(f"算法 {actual_key} 与 blind freeze manifest 不一致")
    dynamic = algorithm.get("dynamic_threshold_source")
    if not isinstance(dynamic, Mapping):
        raise ValueError("算法结果缺少 dynamic threshold 身份")
    if dynamic.get("sha256") != frozen.get("second_round_report_sha256"):
        raise ValueError("dynamic threshold report SHA 与 blind freeze manifest 不一致")
    if not math.isclose(
        float(dynamic.get("validation_q90")),
        float(frozen.get("validation_q90")),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("validation q90 与 blind freeze manifest 不一致")


def verify_processed_video_identities(
    video_summaries: list[Mapping[str, Any]],
    freeze_state: Mapping[str, Any],
) -> None:
    blind_inputs = freeze_state.get("blind_inputs")
    frozen_videos = (
        blind_inputs.get("videos") if isinstance(blind_inputs, Mapping) else None
    )
    if not isinstance(frozen_videos, Mapping):
        raise ValueError("freeze state 缺少 B1-B7 输入身份")
    by_id = {str(summary.get("video_id")): summary for summary in video_summaries}
    if set(by_id) != set(frozen_videos):
        raise ValueError("处理后的视频集合与 freeze manifest 不一致")
    sealed_video_paths = freeze_state.get("sealed_video_paths")
    if sealed_video_paths is not None and (
        not isinstance(sealed_video_paths, Mapping)
        or set(sealed_video_paths) != set(frozen_videos)
    ):
        raise ValueError("freeze state 的 sealed video 集合不完整")
    for video_id, record in frozen_videos.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"freeze state 的 {video_id} 输入记录非法")
        summary = by_id[str(video_id)]
        frozen_path = _resolve_manifest_path(record)
        processed_path = Path(str(summary.get("video_path"))).resolve()
        expected_processed_path = (
            Path(str(sealed_video_paths[video_id])).resolve()
            if isinstance(sealed_video_paths, Mapping)
            else frozen_path
        )
        if processed_path != expected_processed_path:
            raise ValueError(f"{video_id} 未从冻结的内容寻址副本解码")
        if str(summary.get("video_sha256", "")).lower() != str(
            record.get("sha256", "")
        ).lower():
            raise ValueError(f"{video_id} 在解码期间发生 SHA-256 漂移")
        if int(summary.get("video_bytes", -1)) != int(record.get("bytes", -2)):
            raise ValueError(f"{video_id} 在解码期间发生字节数漂移")
        baseline_path = Path(str(summary.get("baseline_jsonl_path", ""))).resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(f"{video_id} baseline JSONL 不存在：{baseline_path}")
        if file_sha256(baseline_path) != str(
            summary.get("baseline_jsonl_sha256", "")
        ).lower():
            raise ValueError(f"{video_id} baseline JSONL SHA-256 漂移")
def reserve_blind_attempt(
    *,
    receipt_path: Path,
    freeze_state: Mapping[str, Any],
    video_summaries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """输入门通过后、模型指标计算前原子占用本次正式盲测机会。"""

    verify_processed_video_identities(video_summaries, freeze_state)
    canonical_receipt_path = default_attempt_receipt_path(
        str(freeze_state.get("attempt_token", ""))
    )
    if receipt_path.resolve() != canonical_receipt_path:
        raise ValueError(
            "正式盲测 receipt 路径不可覆盖；固定路径为 "
            f"{canonical_receipt_path}"
        )
    payload = {
        "schema_version": BLIND_ATTEMPT_RECEIPT_SCHEMA_VERSION,
        "status": "reserved",
        "reserved_at_utc": utc_now_iso(),
        "completed_at_utc": None,
        "attempt_token": freeze_state["attempt_token"],
        "attempt_identity_sha256": freeze_state["attempt_identity_sha256"],
        "freeze_manifest_path": freeze_state["path"],
        "freeze_manifest_sha256": freeze_state["sha256"],
        "git_revision": freeze_state["git"]["revision"],
        "videos": [
            {
                "video_id": summary["video_id"],
                "video_path": summary["video_path"],
                "video_sha256": summary["video_sha256"],
                "video_bytes": summary["video_bytes"],
                "baseline_jsonl_path": summary["baseline_jsonl_path"],
                "baseline_jsonl_sha256": summary["baseline_jsonl_sha256"],
            }
            for summary in video_summaries
        ],
        "report": None,
        "error": None,
    }
    try:
        _atomic_create_json(canonical_receipt_path, payload)
    except FileExistsError as exc:
        raise RuntimeError(
            f"正式盲测 receipt 已存在，拒绝重复计算算法结果：{canonical_receipt_path}"
        ) from exc
    return payload


def finalize_blind_attempt(
    receipt_path: Path,
    *,
    status: str,
    report_path: Path | None = None,
    error: str | None = None,
) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("blind attempt 最终状态只能是 completed 或 failed")
    path = receipt_path.resolve()
    receipt = read_json_object(path)
    if receipt.get("schema_version") != BLIND_ATTEMPT_RECEIPT_SCHEMA_VERSION:
        raise ValueError("blind attempt receipt schema_version 不受支持")
    if path != default_attempt_receipt_path(str(receipt.get("attempt_token", ""))):
        raise ValueError("blind attempt receipt 不在确定性固定路径")
    if receipt.get("attempt_identity_sha256") != receipt.get("attempt_token"):
        raise ValueError("blind attempt receipt 身份不一致")
    if receipt.get("status") != "reserved":
        raise ValueError("blind attempt receipt 已经冻结，不能重复结束")
    receipt["status"] = status
    receipt["completed_at_utc"] = utc_now_iso()
    receipt["error"] = error
    if report_path is not None and report_path.is_file():
        receipt["report"] = {
            "path": str(report_path.resolve()),
            "sha256": file_sha256(report_path.resolve()),
            "bytes": report_path.stat().st_size,
        }
    _atomic_write_json(path, receipt)
