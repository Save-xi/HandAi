from pathlib import Path

from utils.config import load_config


def test_load_config_resolves_from_repo_root(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    monkeypatch.chdir(repo_root)

    cfg = load_config("configs/default.yaml")

    assert Path(cfg["output_json_path"]) == project_root / "examples" / "sample_output.json"
    assert Path(cfg["jsonl_output_dir"]) == project_root / "outputs"
    assert cfg["enable_control_extension"] is False
    assert cfg["svh_enable_preview"] is False
    assert cfg["gui_enabled"] is True
    assert cfg["headless"] is False


def test_load_svh_9ch_preview_config_resolves_example_output(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    monkeypatch.chdir(repo_root)

    cfg = load_config("configs/svh_9ch_preview.yaml")

    assert cfg["svh_preview_layout"] == "svh_9ch"
    assert cfg["svh_preview_channel_count"] == 9
    assert cfg["enable_control_extension"] is True
    assert cfg["svh_enable_preview"] is True
    assert Path(cfg["output_json_path"]) == project_root / "examples" / "sample_output_svh_9ch.json"
