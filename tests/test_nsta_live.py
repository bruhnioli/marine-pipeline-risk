"""Live smoke test against the real NSTA ArcGIS service.

Excluded from the default `pytest` run (see `-m "not live"` in
pyproject.toml) so the normal suite never depends on NSTA availability.
Run explicitly with:

    uv run pytest -m live
"""

from pathlib import Path

import pytest

from marine_engine.providers import nsta

pytestmark = pytest.mark.live


def test_pl854_is_reachable_and_resolvable(tmp_path: Path):
    result = nsta.fetch_pipeline("PL854", cache_dir=tmp_path)

    assert result.source_label in ("active", "removed")
    assert result.feature["properties"]["NSTAPIPNO"] == "PL854"

    geometry = nsta.geometry_from_feature(result.feature)
    assert geometry.geom_type in ("LineString", "MultiLineString")
