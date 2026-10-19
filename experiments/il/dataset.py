"""Corpus loading utilities for imitation-learning experiments."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class CorpusError(ValueError):
    """Raised when an IL corpus is malformed or violates split invariants."""


_REQUIRED = {
    "policy_observation", "condition", "split", "scenario_id", "course_family", "label"
}
_LABEL_REQUIRED = {"feasible", "offset_mm", "accel_mm_s2", "certificate_path", "certificate_ok"}
_CONDITIONS = {"optimal", "approx", "heuristic", "cert_control"}
_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class DatasetArrays:
    """Numerical training data and provenance for a selected corpus subset."""

    x: np.ndarray
    y: np.ndarray  # columns: offset (m), acceleration (m/s^2)
    group_ids: np.ndarray
    metadata: tuple[dict[str, Any], ...]
    grid: np.ndarray
    dropped_infeasible: int = 0

    @property
    def dropped_count(self) -> int:
        return self.dropped_infeasible

    def as_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "group_ids": self.group_ids,
                "metadata": self.metadata, "grid": self.grid,
                "dropped_infeasible": self.dropped_infeasible}


def validate_corpus(corpus: str | Path) -> Path:
    """Validate corpus files and ensure no scenario crosses split boundaries."""
    root = Path(corpus)
    if not root.is_dir():
        raise CorpusError(f"corpus directory does not exist: {root}")
    for name in ("manifest.json", "examples.jsonl"):
        p = root / name
        if not p.is_file():
            raise CorpusError(f"corpus missing required file: {name}")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"invalid manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CorpusError("manifest.json must contain a JSON object")

    scenario_splits: dict[str, str] = {}
    try:
        with (root / "examples.jsonl").open(encoding="utf-8") as stream:
            for lineno, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise CorpusError(f"examples.jsonl:{lineno}: invalid JSON: {exc}") from exc
                _validate_record(record, lineno)
                sid, split = record["scenario_id"], record["split"]
                prior = scenario_splits.setdefault(str(sid), split)
                if prior != split:
                    raise CorpusError(f"scenario {sid!r} occurs in both {prior!r} and {split!r} splits")
    except OSError as exc:
        raise CorpusError(f"cannot read examples.jsonl: {exc}") from exc
    return root


def _validate_record(record: Any, lineno: int) -> None:
    if not isinstance(record, dict):
        raise CorpusError(f"examples.jsonl:{lineno}: record must be an object")
    missing = _REQUIRED - set(record)
    if missing:
        raise CorpusError(f"examples.jsonl:{lineno}: missing fields {sorted(missing)}")
    if not isinstance(record["condition"], str) or record["condition"] not in _CONDITIONS:
        raise CorpusError(f"examples.jsonl:{lineno}: invalid condition {record['condition']!r}")
    if record["split"] not in _SPLITS:
        raise CorpusError(f"examples.jsonl:{lineno}: invalid split {record['split']!r}")
    if not isinstance(record["label"], dict):
        raise CorpusError(f"examples.jsonl:{lineno}: label must be an object")
    missing = _LABEL_REQUIRED - set(record["label"])
    if missing:
        raise CorpusError(f"examples.jsonl:{lineno}: missing label fields {sorted(missing)}")


def load_examples(corpus: str | Path, *, condition: str | Iterable[str] | None = None,
                  split: str | Iterable[str] | None = None, drop_infeasible: bool = True,
                  return_stats: bool = False) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load records, filtering condition/split; optionally return drop statistics."""
    root = validate_corpus(corpus)
    conditions = {condition} if isinstance(condition, str) else (set(condition) if condition is not None else None)
    splits = {split} if isinstance(split, str) else (set(split) if split is not None else None)
    if conditions is not None and not conditions <= _CONDITIONS:
        raise CorpusError(f"invalid condition filter: {sorted(conditions - _CONDITIONS)}")
    if splits is not None and not splits <= _SPLITS:
        raise CorpusError(f"invalid split filter: {sorted(splits - _SPLITS)}")
    selected, dropped = [], 0
    with (root / "examples.jsonl").open(encoding="utf-8") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            record = json.loads(raw)
            if conditions is not None and record["condition"] not in conditions:
                continue
            if splits is not None and record["split"] not in splits:
                continue
            if drop_infeasible and not bool(record["label"]["feasible"]):
                dropped += 1
                continue
            selected.append(record)
    stats = {"dropped_infeasible": dropped, "count": len(selected), "corpus": str(root)}
    return (selected, stats) if return_stats else selected


def to_arrays(records: Iterable[Mapping[str, Any]], *, dropped_infeasible: int = 0) -> DatasetArrays:
    """Convert records to features and metric targets (metres and m/s²)."""
    records = list(records)
    grid = np.arange(0, 11, 1, dtype=float)
    try:
        from autocross.datagen.dataset import policy_features
    except ImportError as exc:
        raise CorpusError("autocross policy_features is unavailable") from exc
    x = np.stack([policy_features(r["policy_observation"], grid) for r in records], axis=0) if records else np.empty((0, 0))
    y = np.asarray([[float(r["label"]["offset_mm"]) / 1000.0,
                     float(r["label"]["accel_mm_s2"]) / 1000.0] for r in records], dtype=float).reshape((-1, 2))
    groups = np.asarray([str(r["scenario_id"]) for r in records], dtype=str)
    metadata = tuple({k: r[k] for k in ("condition", "split", "scenario_id", "course_family")} for r in records)
    return DatasetArrays(x, y, groups, metadata, grid, dropped_infeasible)


def load_dataset(corpus: str | Path, **kwargs: Any) -> DatasetArrays:
    records, stats = load_examples(corpus, return_stats=True, **kwargs)
    return to_arrays(records, dropped_infeasible=stats["dropped_infeasible"])
