"""Tests for marine_engine.config."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from marine_engine.config import StudyConfig, load_study_config


def test_pl854_config_loads(pl854_config_path: Path) -> None:
    config = load_study_config(pl854_config_path)

    assert config.study.id == "PL854"
    assert config.crs.horizontal == "EPSG:32631"
    assert config.paths.raw_dir == Path("data/raw")
    assert config.pipeline["pipeline_id"] == "PL854"
    assert config.area_of_interest.corridor_buffer_m == 5000
    assert config.pipeline["chainage_interval_m"] == 25


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        yaml.safe_dump(
            {
                "study": {"id": "X", "name": "Test"},
                "crs": {"horizontal": "EPSG:4326"},
                "not_a_real_section": {},
            }
        )
    )

    with pytest.raises(ValidationError):
        load_study_config(bad_config)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.safe_dump({"study": {"id": "X", "name": "Test"}}))

    with pytest.raises(ValidationError):
        load_study_config(bad_config)


def test_area_of_interest_defaults_to_unset() -> None:
    config = StudyConfig.model_validate(
        {
            "study": {"id": "X", "name": "Test"},
            "crs": {"horizontal": "EPSG:4326"},
        }
    )

    assert config.area_of_interest.bbox_wgs84 is None
