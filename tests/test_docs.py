"""Documentation must not drift from the code."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _keys(tree: dict, prefix: str = "") -> list[str]:
    keys = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(_keys(value, path))
        else:
            keys.append(path)
    return keys


def test_every_default_setting_is_documented() -> None:
    config = yaml.safe_load((ROOT / "config" / "default_config.yaml").read_text(encoding="utf-8"))
    text = (DOCS / "configuration.md").read_text(encoding="utf-8")
    missing = []
    for key in _keys(config):
        if key.startswith("regolith.specific_heat.c"):
            continue  # documented together as c0 … c4
        if f"`{key}`" not in text:
            missing.append(key)
    assert not missing, f"undocumented settings: {missing}"


def test_api_examples_run(tmp_path, monkeypatch) -> None:
    text = (DOCS / "api.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) >= 3
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, "docs/api.md", "exec"), namespace)  # noqa: S102


def test_site_builds_without_broken_links() -> None:
    pytest.importorskip("markdown_it")
    pytest.importorskip("mdit_py_plugins")
    pytest.importorskip("jinja2")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_docs.py"), "--strict"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_runs_from_any_directory(tmp_path) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-m", "simulation.cli", "run", "--duration", "1", "--dt", "900",
         "--cratersize", "1500", "--point-source", "--no-figures", "--output", "smoke"],
        capture_output=True, text=True, cwd=tmp_path, env=env, check=False, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (tmp_path / "smoke" / "surface_T_final.npy").is_file()


def test_cli_rejects_bad_arguments(tmp_path) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-m", "simulation.cli", "run", "--dt", "-5"],
        capture_output=True, text=True, cwd=tmp_path, env=env, check=False,
    )
    assert result.returncode != 0
    assert "must be positive" in result.stderr
