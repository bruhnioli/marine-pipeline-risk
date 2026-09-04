"""Structural smoke tests: every stage package must be importable."""

import importlib

import pytest

STAGE_PACKAGES = [
    "marine_engine.providers",
    "marine_engine.preprocessing",
    "marine_engine.morphology",
    "marine_engine.sediment",
    "marine_engine.metocean",
    "marine_engine.pipeline",
    "marine_engine.risk",
    "marine_engine.validation",
    "marine_engine.export",
]


@pytest.mark.parametrize("module_name", STAGE_PACKAGES)
def test_stage_package_imports(module_name: str) -> None:
    importlib.import_module(module_name)
