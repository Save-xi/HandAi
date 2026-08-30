from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = PROJECT_ROOT / "integrations" / "unity_phase15_snapshot"
MANIFEST_PATH = SNAPSHOT_ROOT / "snapshot_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_unity_snapshot_files_match_manifest() -> None:
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["scope"].startswith("single-right-hand")
    assert manifest["batch_acceptance_marker"] == "PHASE15_UNITY_SAFETY_BATCH_PASS"
    assert manifest["files"]

    declared_paths = set()
    for entry in manifest["files"]:
        snapshot_path = SNAPSHOT_ROOT / Path(entry["snapshot_path"])
        declared_paths.add(snapshot_path.resolve())
        assert snapshot_path.is_file(), entry["snapshot_path"]
        assert snapshot_path.stat().st_size == entry["bytes"]
        assert _sha256(snapshot_path) == entry["sha256"]

    actual_paths = {
        path.resolve()
        for path in SNAPSHOT_ROOT.rglob("*")
        if path.is_file()
        and path.name not in {".gitattributes", "README.md", "snapshot_manifest.json"}
    }
    assert actual_paths == declared_paths


def test_local_unity_sources_have_not_drifted_when_available() -> None:
    manifest = _load_manifest()
    unity_root = Path(manifest["unity_project_source"])
    if not unity_root.is_dir():
        return

    for entry in manifest["files"]:
        source_path = unity_root / Path(entry["source_relative_path"])
        assert source_path.is_file(), entry["source_relative_path"]
        assert source_path.stat().st_size == entry["bytes"]
        assert _sha256(source_path) == entry["sha256"], (
            f"Unity source drifted from tracked snapshot: {entry['source_relative_path']}"
        )
