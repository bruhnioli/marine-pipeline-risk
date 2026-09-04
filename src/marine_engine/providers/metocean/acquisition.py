"""Copernicus Marine acquisition bookkeeping: chunking, manifest, resumability (MAR-009).

Deliberately separate from `copernicus.py` (which does the actual Toolbox
calls): this module decides WHAT to request (time chunks, idempotency) and
records WHAT WAS acquired, but never talks to Copernicus Marine itself.

Chunking (Section 15)
-----------------------
A multi-year download is never requested as one giant fragile call.
`generate_monthly_chunks`/`generate_yearly_chunks` split a `[start, end)`
window into UTC-calendar-aligned chunks; each chunk is acquired and
manifested independently, so an interrupted multi-year run resumes from
the manifest rather than restarting.

Idempotency (Section 15)
--------------------------
A chunk is considered already-acquired only when its `product_id`,
`dataset_id`, `variables`, `requested_bbox`, `requested_depths`,
`requested_start`, and `requested_end` all match a manifest entry AND that
entry's local file still exists on disk -- never on time range alone,
since a re-run with different variables/bbox/depths must not be silently
skipped.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MANIFEST_ENTRY_FIELDS = (
    "provider",
    "product_id",
    "dataset_id",
    "evidence_role",
    "variables",
    "requested_bbox",
    "requested_depths",
    "requested_start",
    "requested_end",
    "actual_start",
    "actual_end",
    "temporal_resolution",
    "local_path",
    "file_size_bytes",
    "sha256",
    "download_timestamp",
    "toolbox_version",
    "licence",
)

# Project data-QA policy (Section 6), not a physical constant: how far
# behind the operational forecast/NRT boundary the historical cutoff sits.
CUTOFF_BUFFER_HOURS = 48


def compute_historical_cutoff(now_utc: datetime) -> datetime:
    """The latest complete UTC day at least `CUTOFF_BUFFER_HOURS` before `now_utc`.

    Returns the EXCLUSIVE end of the historical window: midnight UTC of the
    calendar day containing `now_utc - CUTOFF_BUFFER_HOURS`, so every
    included hourly timestamp belongs to a day that is itself entirely at
    least `CUTOFF_BUFFER_HOURS` old. Never includes a forecast timestamp.
    """

    safe_moment = now_utc - timedelta(hours=CUTOFF_BUFFER_HOURS)
    return datetime(safe_moment.year, safe_moment.month, safe_moment.day, tzinfo=UTC)


def generate_monthly_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """UTC-calendar-month-aligned (chunk_start, chunk_end) pairs spanning `[start, end)`."""

    if start >= end:
        return []

    chunks: list[tuple[datetime, datetime]] = []
    cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
    while cursor < end:
        next_month = cursor.month + 1
        next_year = cursor.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        chunk_end = datetime(next_year, next_month, 1, tzinfo=UTC)
        chunk_start = max(cursor, start)
        chunks.append((chunk_start, min(chunk_end, end)))
        cursor = chunk_end
    return chunks


def generate_yearly_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """UTC-calendar-year-aligned (chunk_start, chunk_end) pairs spanning `[start, end)`."""

    if start >= end:
        return []

    chunks: list[tuple[datetime, datetime]] = []
    cursor = datetime(start.year, 1, 1, tzinfo=UTC)
    while cursor < end:
        chunk_end = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        chunk_start = max(cursor, start)
        chunks.append((chunk_start, min(chunk_end, end)))
        cursor = chunk_end
    return chunks


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


def _identity_matches(
    entry: dict[str, Any],
    *,
    product_id: str,
    dataset_id: str,
    variables: list[str],
    requested_bbox: list[float],
    requested_depths: list[float] | None,
    requested_start: str,
    requested_end: str,
) -> bool:
    return (
        entry.get("product_id") == product_id
        and entry.get("dataset_id") == dataset_id
        and entry.get("variables") == list(variables)
        and entry.get("requested_bbox") == list(requested_bbox)
        and entry.get("requested_depths") == (list(requested_depths) if requested_depths else None)
        and entry.get("requested_start") == requested_start
        and entry.get("requested_end") == requested_end
    )


def find_matching_entry(
    entries: list[dict[str, Any]],
    *,
    product_id: str,
    dataset_id: str,
    variables: list[str],
    requested_bbox: list[float],
    requested_depths: list[float] | None,
    requested_start: str,
    requested_end: str,
) -> dict[str, Any] | None:
    for entry in entries:
        if _identity_matches(
            entry,
            product_id=product_id,
            dataset_id=dataset_id,
            variables=variables,
            requested_bbox=requested_bbox,
            requested_depths=requested_depths,
            requested_start=requested_start,
            requested_end=requested_end,
        ):
            return entry
    return None


def already_acquired(
    manifest_path: Path,
    *,
    product_id: str,
    dataset_id: str,
    variables: list[str],
    requested_bbox: list[float],
    requested_depths: list[float] | None,
    requested_start: str,
    requested_end: str,
) -> dict[str, Any] | None:
    """The existing manifest entry for this exact chunk request, if its file still exists.

    Lets a resumed multi-year acquisition skip chunks it already has --
    never on time range alone, always on the full identity tuple
    (Section 15).
    """

    entries = load_manifest(manifest_path)
    entry = find_matching_entry(
        entries,
        product_id=product_id,
        dataset_id=dataset_id,
        variables=variables,
        requested_bbox=requested_bbox,
        requested_depths=requested_depths,
        requested_start=requested_start,
        requested_end=requested_end,
    )
    if entry is not None and Path(entry["local_path"]).exists():
        return entry
    return None


STATIC_TIME_SENTINEL = "static"


def isoformat_or_static_sentinel(moment: datetime | None) -> str:
    """A static (time-invariant) dataset has no request time range at all.

    Uses a fixed, clearly-labelled sentinel (`STATIC_TIME_SENTINEL`) rather
    than fabricating a start/end timestamp for it -- callers looking up an
    existing static-dataset entry via `already_acquired`/`find_matching_entry`
    should pass this same sentinel for `requested_start`/`requested_end`.
    """

    return moment.isoformat() if moment is not None else STATIC_TIME_SENTINEL


def record_acquisition(
    manifest_path: Path,
    *,
    provider: str,
    product_id: str,
    dataset_id: str,
    evidence_role: str,
    variables: list[str],
    requested_bbox: list[float],
    requested_depths: list[float] | None,
    requested_start: datetime | None,
    requested_end: datetime | None,
    actual_start: str | None,
    actual_end: str | None,
    temporal_resolution: str,
    local_path: Path,
    toolbox_version: str,
    licence: str | None,
    downloaded_at: datetime,
) -> dict[str, Any]:
    """Checksum an already-downloaded chunk and upsert its manifest entry.

    Idempotent by (product_id, dataset_id, variables, requested_bbox,
    requested_depths, requested_start, requested_end): re-running for the
    same identity replaces that one entry rather than appending a
    duplicate. Never stores credentials. `requested_start`/`requested_end`
    are `None` for a static (time-invariant) dataset -- recorded as the
    `"static"` sentinel, never a fabricated timestamp.
    """

    requested_start_str = isoformat_or_static_sentinel(requested_start)
    requested_end_str = isoformat_or_static_sentinel(requested_end)

    entry = {
        "provider": provider,
        "product_id": product_id,
        "dataset_id": dataset_id,
        "evidence_role": evidence_role,
        "variables": list(variables),
        "requested_bbox": list(requested_bbox),
        "requested_depths": list(requested_depths) if requested_depths else None,
        "requested_start": requested_start_str,
        "requested_end": requested_end_str,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "temporal_resolution": temporal_resolution,
        "local_path": str(local_path),
        "file_size_bytes": local_path.stat().st_size,
        "sha256": compute_sha256(local_path),
        "download_timestamp": downloaded_at.isoformat(),
        "toolbox_version": toolbox_version,
        "licence": licence,
    }

    entries = load_manifest(manifest_path)
    entries = [
        e
        for e in entries
        if not _identity_matches(
            e,
            product_id=product_id,
            dataset_id=dataset_id,
            variables=variables,
            requested_bbox=requested_bbox,
            requested_depths=requested_depths,
            requested_start=requested_start_str,
            requested_end=requested_end_str,
        )
    ]
    entries.append(entry)
    save_manifest(manifest_path, entries)
    return entry
