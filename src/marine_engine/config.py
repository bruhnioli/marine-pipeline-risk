"""Configuration schema and loader for study-specific YAML configs.

Each study (e.g. PL854) is described by a single YAML file under `configs/`.
This module defines the schema for that file and a loader that validates a
YAML file into a `StudyConfig` instance. The per-stage sections (`providers`,
`morphology`, `sediment`, ...) are left as open dictionaries here: their
schemas will be defined by the modules that consume them once those stages
are implemented.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StudyInfo(BaseModel):
    """Identity and scope of the study."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    design_life_years: float | None = None


class CrsConfig(BaseModel):
    """Coordinate reference systems used for processed geometries and rasters."""

    model_config = ConfigDict(extra="forbid")

    horizontal: str
    vertical: str | None = None


class AreaOfInterest(BaseModel):
    """Spatial extent of the study corridor.

    Left unset until real pipeline route / bathymetry data is ingested.
    """

    model_config = ConfigDict(extra="forbid")

    bbox_wgs84: list[float] | None = None
    corridor_buffer_m: float | None = None


class DataPaths(BaseModel):
    """Locations of the local data lake stages, relative to the project root."""

    model_config = ConfigDict(extra="forbid")

    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")


class StudyConfig(BaseModel):
    """Root schema for a study-specific YAML configuration file."""

    model_config = ConfigDict(extra="forbid")

    study: StudyInfo
    crs: CrsConfig
    area_of_interest: AreaOfInterest = Field(default_factory=AreaOfInterest)
    paths: DataPaths = Field(default_factory=DataPaths)

    providers: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    morphology: dict[str, Any] = Field(default_factory=dict)
    sediment: dict[str, Any] = Field(default_factory=dict)
    metocean: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)


def load_study_config(path: str | Path) -> StudyConfig:
    """Load and validate a study configuration YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return StudyConfig.model_validate(raw)
