"""Raw bathymetry acquisition bookkeeping: paths, checksums, idempotent manifest.

Deliberately source-agnostic: `ukho.py`/`bgs.py`/`emodnet.py` do the actual
downloading, then hand the resulting local file to `record_acquisition` here
to compute its checksum and upsert a manifest entry. Never alters a
downloaded file's bytes.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

MANIFEST_ENTRY_FIELDS = (
    "source",
    "dataset_id",
    "source_url_or_service",
    "acquisition_timestamp",
    "request_parameters",
    "original_filename",
    "local_path",
    "file_size_bytes",
    "sha256",
    "licence",
    "acquisition_year",
    "horizontal_crs",
    "vertical_datum",
    "nominal_resolution_m",
    "raw_unmodified",
)


def raw_dataset_dir(raw_dir: Path, source: str, dataset_id: str) -> Path:
    """The canonical raw-storage directory for one source's one dataset/survey."""

    return raw_dir / "bathymetry" / source.lower() / dataset_id


def compute_sha256(path: Path) -> str:
    """Stream a file through SHA-256 without loading it entirely into memory."""

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


def find_matching_entry(
    entries: list[dict[str, Any]], source: str, dataset_id: str, request_parameters: dict[str, Any]
) -> dict[str, Any] | None:
    """Find a manifest entry for the same source/dataset/request, if one exists."""

    for entry in entries:
        if (
            entry.get("source") == source
            and entry.get("dataset_id") == dataset_id
            and entry.get("request_parameters") == request_parameters
        ):
            return entry
    return None


def already_acquired(
    manifest_path: Path, source: str, dataset_id: str, request_parameters: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the existing manifest entry if this exact acquisition was already made
    and its file is still present on disk -- lets callers skip a redundant download.
    """

    entries = load_manifest(manifest_path)
    entry = find_matching_entry(entries, source, dataset_id, request_parameters)
    if entry is not None and Path(entry["local_path"]).exists():
        return entry
    return None


def record_acquisition(
    manifest_path: Path,
    *,
    source: str,
    dataset_id: str,
    source_url_or_service: str,
    request_parameters: dict[str, Any],
    local_path: Path,
    licence: str | None,
    acquisition_year: int | None,
    horizontal_crs: str | None,
    vertical_datum: str | None,
    nominal_resolution_m: float | None,
    acquired_at: datetime,
) -> dict[str, Any]:
    """Checksum an already-downloaded file and upsert its manifest entry.

    Idempotent by (source, dataset_id): re-running for the same identifiers
    replaces that one entry rather than appending a duplicate. A different
    dataset_id (a different survey/epoch) always gets its own entry and,
    via `raw_dataset_dir`, its own file path -- never overwriting another
    epoch's file.
    """

    entry = {
        "source": source,
        "dataset_id": dataset_id,
        "source_url_or_service": source_url_or_service,
        "acquisition_timestamp": acquired_at.isoformat(),
        "request_parameters": request_parameters,
        "original_filename": local_path.name,
        "local_path": str(local_path),
        "file_size_bytes": local_path.stat().st_size,
        "sha256": compute_sha256(local_path),
        "licence": licence,
        "acquisition_year": acquisition_year,
        "horizontal_crs": horizontal_crs,
        "vertical_datum": vertical_datum,
        "nominal_resolution_m": nominal_resolution_m,
        "raw_unmodified": True,
    }

    entries = load_manifest(manifest_path)
    entries = [
        e for e in entries if not (e.get("source") == source and e.get("dataset_id") == dataset_id)
    ]
    entries.append(entry)
    save_manifest(manifest_path, entries)
    return entry
