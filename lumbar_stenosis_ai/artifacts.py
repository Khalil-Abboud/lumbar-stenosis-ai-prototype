"""Reproducible, privacy-conscious persistence for trained research models."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from lumbar_stenosis_ai import __version__
from lumbar_stenosis_ai.config import PipelineConfig
from lumbar_stenosis_ai.models import FuzzyARTMAPClassifier


BUNDLE_VERSION = 2

SNAPSHOT_FIELDS = (
    "sample_id",
    "patient_hash",
    "image_sha256",
    "label",
    "partition",
    "plane",
    "level",
    "sequence",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return destination


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "lumbar_stenosis_ai": __version__}
    for package in ("numpy", "pillow", "torch", "torchvision", "pydicom"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a medical image into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_patient_id(patient_id: str, salt: str) -> str:
    """Create a run-local token without persisting a raw patient identifier."""

    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(patient_id.encode("utf-8"))
    return digest.hexdigest()


def save_manifest_snapshot(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> Path:
    """Persist a deidentified, path-free input snapshot for reproducibility."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SNAPSHOT_FIELDS})
    return destination


def load_manifest_snapshot(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Deidentified manifest snapshot not found: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SNAPSHOT_FIELDS:
            raise ValueError("Unsupported deidentified manifest snapshot schema.")
        return [dict(row) for row in reader]


def save_model_bundle(
    output_dir: str | Path,
    *,
    model: FuzzyARTMAPClassifier,
    config: PipelineConfig,
    feature_extractor: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> Path:
    """Save ART-MAP arrays separately from human-readable metadata."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state = model.to_dict()
    arrays_path = directory / "model_state.npz"
    np.savez_compressed(
        arrays_path,
        weights=np.asarray(state["state"]["weights"], dtype=np.float64),
        category_counts=np.asarray(state["state"]["category_counts"], dtype=np.int64),
        scaler_min=np.asarray(state["scaler"]["feature_min"], dtype=np.float64),
        scaler_max=np.asarray(state["scaler"]["feature_max"], dtype=np.float64),
    )
    metadata_payload = {
        "bundle_version": BUNDLE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_use_only": True,
        "config": config.to_dict(),
        "software_versions": package_versions(),
        "feature_extractor": dict(feature_extractor),
        "model": {
            "name": state["model"],
            "version": state["version"],
            "parameters": state["parameters"],
            "n_features_in": state["state"]["n_features_in"],
            "category_labels": state["state"]["category_labels"],
            "category_count": len(state["state"]["category_labels"]),
            "scaler_clip": state["scaler"]["clip"],
            "arrays_file": arrays_path.name,
            "arrays_sha256": sha256_file(arrays_path),
        },
        "run": dict(run_metadata),
    }
    write_json(directory / "metadata.json", metadata_payload)
    return directory


def load_model_bundle(
    model_dir: str | Path,
) -> tuple[FuzzyARTMAPClassifier, PipelineConfig, dict[str, Any]]:
    directory = Path(model_dir)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Unsupported model bundle version.")
    model_metadata = payload["model"]
    arrays_path = directory / model_metadata["arrays_file"]
    if sha256_file(arrays_path) != model_metadata.get("arrays_sha256"):
        raise ValueError("Model array file failed its integrity check.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        state = {
            "model": model_metadata["name"],
            "version": model_metadata["version"],
            "parameters": model_metadata["parameters"],
            "scaler": {
                "clip": model_metadata["scaler_clip"],
                "feature_min": arrays["scaler_min"].tolist(),
                "feature_max": arrays["scaler_max"].tolist(),
            },
            "state": {
                "n_features_in": model_metadata["n_features_in"],
                "weights": arrays["weights"].tolist(),
                "category_labels": model_metadata["category_labels"],
                "category_counts": arrays["category_counts"].tolist(),
            },
        }
    model = FuzzyARTMAPClassifier.from_dict(state)
    return model, PipelineConfig.from_dict(payload["config"]), payload


def save_feature_matrix(
    path: str | Path, *, sample_ids: Iterable[str], features: np.ndarray
) -> Path:
    destination = Path(path)
    if destination.suffix.casefold() != ".npz":
        destination = Path(f"{destination}.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Passing a filename directly makes NumPy append '.npz' implicitly.  A
    # file handle keeps the returned path exactly equal to the written path.
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_ids=np.asarray(list(sample_ids), dtype=str),
            features=np.asarray(features, dtype=np.float32),
        )
    return destination


def save_predictions(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "true_label",
        "predicted_label",
        "category",
        "choice",
        "match",
        "meets_vigilance",
        "rejected",
        "is_correct",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return destination


def save_patient_predictions(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> Path:
    """Save deidentified patient-aggregated evaluation rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_hash",
        "true_label",
        "predicted_label",
        "slice_count",
        "is_correct",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return destination
