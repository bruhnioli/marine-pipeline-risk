"""Shared pytest fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def pl854_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "pl854.yaml"
