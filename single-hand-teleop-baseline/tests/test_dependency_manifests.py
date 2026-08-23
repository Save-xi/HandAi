from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalized_pin(value: str) -> str:
    return value.strip().lower()


def test_conda_pip_dependencies_match_requirements_txt():
    requirements = {
        _normalized_pin(line)
        for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    environment = yaml.safe_load((PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8"))
    pip_dependencies: list[str] = []
    for dependency in environment["dependencies"]:
        if isinstance(dependency, dict) and "pip" in dependency:
            pip_dependencies.extend(dependency["pip"])

    assert {_normalized_pin(value) for value in pip_dependencies} == requirements
