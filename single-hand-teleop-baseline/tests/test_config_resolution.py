from pathlib import Path

from utils.config import load_config


def test_load_config_resolves_from_repo_root(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    monkeypatch.chdir(repo_root)

    cfg = load_config("configs/default.yaml")

    assert Path(cfg["output_json_path"]) == project_root / "examples" / "sample_output.json"
