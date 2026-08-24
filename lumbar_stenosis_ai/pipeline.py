"""End-to-end training and prediction orchestration."""

from __future__ import annotations

import secrets
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lumbar_stenosis_ai.artifacts import (
    hash_patient_id,
    load_manifest_snapshot,
    load_model_bundle,
    save_feature_matrix,
    save_manifest_snapshot,
    save_model_bundle,
    save_patient_predictions,
    save_predictions,
    sha256_file,
    write_json,
)
from lumbar_stenosis_ai.config import PipelineConfig
from lumbar_stenosis_ai.data.manifest import ManifestRecord, load_manifest
from lumbar_stenosis_ai.evaluation import classification_metrics
from lumbar_stenosis_ai.features import ResNet18FeatureExtractor
from lumbar_stenosis_ai.models import FuzzyARTMAPClassifier
from lumbar_stenosis_ai.splitting import patient_level_split


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_dir: Path
    train_count: int
    test_count: int
    category_count: int
    metrics: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class UpdateResult:
    output_dir: Path
    sample_count: int
    previous_category_count: int
    updated_category_count: int
    out_of_training_range_value_count: int


def _prepare_output_directory(path: str | Path) -> Path:
    """Create a new run directory, refusing to mix stale experiment files."""

    destination = Path(path).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(
                "Output directory must be new or empty to preserve experiment "
                f"integrity: {destination}"
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def _experiment_stratum(records: Sequence[ManifestRecord]) -> dict[str, str]:
    """Require one explicit MRI stratum instead of silently pooling anatomy."""

    result: dict[str, str] = {}
    for field in ("plane", "level", "sequence"):
        values = {
            str(getattr(record, field, "")).strip().casefold()
            for record in records
            if str(getattr(record, field, "")).strip()
        }
        if any(not str(getattr(record, field, "")).strip() for record in records):
            raise ValueError(
                f"Every training row needs a {field} value so anatomical "
                "experiments cannot be pooled silently."
            )
        if len(values) != 1:
            raise ValueError(
                f"Train one model per {field}; this manifest contains: "
                + ", ".join(sorted(values))
            )
        result[field] = next(iter(values))
    return result


def _extractor_metadata(extractor: Any) -> dict[str, Any]:
    try:
        metadata = dict(extractor.metadata())
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "The feature extractor must expose reproducibility metadata."
        ) from exc
    required = {"weights_id", "preprocessing_id", "state_sha256"}
    if not required.issubset(metadata):
        raise RuntimeError("Feature-extractor metadata is incomplete.")
    return metadata


def _assert_extractor_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key in ("name", "output_features", "weights_id", "preprocessing_id", "state_sha256"):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(
                "The reconstructed ResNet18 feature space does not match the "
                f"saved model bundle ({key})."
            )


def _snapshot_rows(
    records: Sequence[ManifestRecord],
    *,
    partition: str,
    patient_hash_salt: str,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": record.sample_id,
            "patient_hash": hash_patient_id(record.patient_id, patient_hash_salt),
            "image_sha256": sha256_file(record.image_path),
            "label": record.label,
            "partition": partition,
            "plane": record.plane,
            "level": record.level,
            "sequence": record.sequence,
        }
        for record in records
    ]


def _distribution(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _assert_unique_image_content(rows: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(str(row["image_sha256"]) for row in rows)
    duplicates = sorted(
        str(row["sample_id"])
        for row in rows
        if counts[str(row["image_sha256"])] > 1
    )
    if duplicates:
        raise ValueError(
            "Exact duplicate image content is not allowed across samples: "
            + ", ".join(duplicates)
        )


def _feature_range_diagnostics(
    model: FuzzyARTMAPClassifier, features: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    scaler = model.scaler_
    assert scaler.feature_min_ is not None
    assert scaler.feature_max_ is not None
    outside = (features < scaler.feature_min_) | (features > scaler.feature_max_)
    per_sample = np.count_nonzero(outside, axis=1)
    value_count = int(np.count_nonzero(outside))
    total_values = int(outside.size)
    return (
        {
            "out_of_training_range_value_count": value_count,
            "total_feature_value_count": total_values,
            "out_of_training_range_fraction": (
                value_count / total_values if total_values else 0.0
            ),
            "affected_sample_count": int(np.count_nonzero(per_sample)),
            "values_are_clipped": bool(model.scaler_.clip),
        },
        per_sample,
    )


def _assert_patient_label_consistency(records: Sequence[ManifestRecord]) -> None:
    labels_by_patient: dict[str, set[str]] = defaultdict(set)
    samples_by_patient: dict[str, list[str]] = defaultdict(list)
    for record in records:
        labels_by_patient[record.patient_id].add(str(record.label))
        samples_by_patient[record.patient_id].append(record.sample_id)
    conflicting_samples = sorted(
        sample_id
        for patient_id, labels in labels_by_patient.items()
        if len(labels) > 1
        for sample_id in samples_by_patient[patient_id]
    )
    if conflicting_samples:
        raise ValueError(
            "One patient has conflicting labels within the same plane/level/sequence; "
            "affected sample IDs: "
            + ", ".join(conflicting_samples)
        )


def _aggregate_patient_predictions(
    records: Sequence[ManifestRecord],
    predicted_labels: Sequence[str],
    *,
    patient_hash_salt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[tuple[ManifestRecord, str]]] = defaultdict(list)
    for record, prediction in zip(records, predicted_labels, strict=True):
        grouped[record.patient_id].append((record, prediction))

    rows: list[dict[str, Any]] = []
    true_patient_labels: list[str] = []
    predicted_patient_labels: list[str] = []
    for patient_id in sorted(grouped):
        items = grouped[patient_id]
        true_labels = {str(record.label) for record, _ in items}
        if len(true_labels) != 1:
            raise ValueError("Patient-level evaluation requires one label per stratum.")
        true_label = next(iter(true_labels))
        vote_counts = Counter(prediction for _, prediction in items)
        ranked_votes = sorted(vote_counts.items(), key=lambda item: (-item[1], item[0]))
        predicted_label = (
            "__no_consensus__"
            if len(ranked_votes) > 1 and ranked_votes[0][1] == ranked_votes[1][1]
            else ranked_votes[0][0]
        )
        rows.append(
            {
                "patient_hash": hash_patient_id(patient_id, patient_hash_salt),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "slice_count": len(items),
                "is_correct": true_label == predicted_label,
            }
        )
        true_patient_labels.append(true_label)
        predicted_patient_labels.append(predicted_label)
    metrics = classification_metrics(true_patient_labels, predicted_patient_labels)
    metrics["aggregation"] = "slice_majority_vote_with_tie_rejection"
    return rows, metrics


def _normalized_split(record: ManifestRecord) -> str:
    return (record.split or "").strip().lower()


def split_manifest_records(
    records: Sequence[ManifestRecord], config: PipelineConfig
) -> tuple[list[ManifestRecord], list[ManifestRecord], str]:
    split_values = {_normalized_split(record) for record in records}
    has_explicit = any(split_values)
    if has_explicit:
        if "" in split_values:
            raise ValueError("Either set split for every row or leave split empty for every row.")
        train = [record for record in records if _normalized_split(record) == "train"]
        test = [record for record in records if _normalized_split(record) == "test"]
        unexpected = split_values - {"train", "test"}
        if unexpected:
            raise ValueError(f"Unsupported split values for training: {sorted(unexpected)}")
        if not train:
            raise ValueError("An explicit manifest needs at least one train row.")
        return train, test, "explicit_patient_split"

    if config.test_fraction == 0:
        return list(records), [], "train_only"
    train, test = patient_level_split(
        records, test_fraction=config.test_fraction, seed=config.seed
    )
    return train, test, "automatic_patient_split"


def _assert_no_patient_leakage(
    train: Sequence[ManifestRecord], test: Sequence[ManifestRecord]
) -> None:
    train_patients = {record.patient_id for record in train}
    test_patients = {record.patient_id for record in test}
    overlap = train_patients & test_patients
    if overlap:
        raise ValueError(
            "Patient leakage detected between training and test partitions: "
            + ", ".join(sorted(overlap))
        )


def train_pipeline(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: PipelineConfig | None = None,
    save_features: bool = True,
) -> TrainingResult:
    """Train ART-MAP on ResNet18 features and evaluate only unseen patients."""

    active_config = config or PipelineConfig()
    records = load_manifest(manifest_path, require_labels=True)
    stratum = _experiment_stratum(records)
    _assert_patient_label_consistency(records)
    train_records, test_records, split_method = split_manifest_records(records, active_config)
    _assert_no_patient_leakage(train_records, test_records)
    unseen_labels = {record.label for record in test_records} - {
        record.label for record in train_records
    }
    if unseen_labels:
        raise ValueError(
            "The test set contains labels absent from training: "
            + ", ".join(sorted(unseen_labels))
        )

    patient_hash_salt = secrets.token_hex(16)
    snapshot_rows = _snapshot_rows(
        train_records,
        partition="train",
        patient_hash_salt=patient_hash_salt,
    ) + _snapshot_rows(
        test_records,
        partition="test",
        patient_hash_salt=patient_hash_salt,
    )
    _assert_unique_image_content(snapshot_rows)

    destination = _prepare_output_directory(output_dir)
    extractor = ResNet18FeatureExtractor(
        pretrained=active_config.pretrained,
        device=active_config.device,
        seed=active_config.seed,
    )
    feature_extractor_metadata = _extractor_metadata(extractor)
    train_extraction = extractor.extract(
        [record.image_path for record in train_records], batch_size=active_config.batch_size
    )
    test_extraction = (
        extractor.extract(
            [record.image_path for record in test_records],
            batch_size=active_config.batch_size,
        )
        if test_records
        else None
    )

    rng = np.random.default_rng(active_config.seed)
    order = rng.permutation(len(train_records))
    train_features = train_extraction.features[order]
    train_labels = np.asarray([record.label for record in train_records], dtype=object)[order]
    training_order_ids = [train_records[index].sample_id for index in order]

    model = FuzzyARTMAPClassifier(
        vigilance=active_config.art_vigilance,
        alpha=active_config.art_choice,
        beta=active_config.art_learning_rate,
        match_tracking_epsilon=active_config.art_match_tracking,
    )
    model.fit(train_features, train_labels)

    metrics: dict[str, Any] | None = None
    if test_records and test_extraction is not None:
        range_diagnostics, _ = _feature_range_diagnostics(
            model, test_extraction.features
        )
        predicted, details = model.predict_with_details(test_extraction.features)
        true_labels = [str(record.label) for record in test_records]
        predicted_labels = [
            "__no_resonance__" if value is None else str(value) for value in predicted
        ]
        metrics = classification_metrics(true_labels, predicted_labels)
        metrics["feature_range_diagnostics"] = range_diagnostics
        patient_rows, patient_metrics = _aggregate_patient_predictions(
            test_records,
            predicted_labels,
            patient_hash_salt=patient_hash_salt,
        )
        metrics["patient_level"] = patient_metrics
        prediction_rows = []
        for record, expected, actual, detail in zip(
            test_records, true_labels, predicted_labels, details, strict=True
        ):
            prediction_rows.append(
                {
                    "sample_id": record.sample_id,
                    "true_label": expected,
                    "predicted_label": actual,
                    **detail,
                    "is_correct": expected == actual,
                }
            )
        save_predictions(destination / "test_predictions.csv", prediction_rows)
        save_patient_predictions(
            destination / "patient_test_predictions.csv", patient_rows
        )
        write_json(destination / "metrics.json", metrics)

    if save_features:
        save_feature_matrix(
            destination / "train_features.npz",
            sample_ids=[record.sample_id for record in train_records],
            features=train_extraction.features,
        )
        if test_records and test_extraction is not None:
            save_feature_matrix(
                destination / "test_features.npz",
                sample_ids=[record.sample_id for record in test_records],
                features=test_extraction.features,
            )

    snapshot_path = save_manifest_snapshot(
        destination / "manifest_snapshot.csv", snapshot_rows
    )

    run_metadata = {
        "run_type": "initial_training",
        "manifest_name": Path(manifest_path).name,
        "split_method": split_method,
        "seed": active_config.seed,
        "experiment_stratum": stratum,
        "train_sample_ids": [record.sample_id for record in train_records],
        "test_sample_ids": [record.sample_id for record in test_records],
        "training_order_sample_ids": training_order_ids,
        "train_patient_count": len({record.patient_id for record in train_records}),
        "test_patient_count": len({record.patient_id for record in test_records}),
        "train_sample_count": len(train_records),
        "test_sample_count": len(test_records),
        "train_class_counts": _distribution(
            [str(record.label) for record in train_records]
        ),
        "test_class_counts": _distribution(
            [str(record.label) for record in test_records]
        ),
        "artmap_category_count": model.n_categories_,
        "has_unseen_test_metrics": metrics is not None,
        "has_patient_level_metrics": metrics is not None,
        "provenance": {
            "snapshot_file": snapshot_path.name,
            "snapshot_sha256": sha256_file(snapshot_path),
            "patient_hash_salt": patient_hash_salt,
            "raw_patient_ids_persisted": False,
            "raw_image_paths_persisted": False,
        },
    }
    save_model_bundle(
        destination,
        model=model,
        config=active_config,
        feature_extractor=feature_extractor_metadata,
        run_metadata=run_metadata,
    )
    return TrainingResult(
        output_dir=destination,
        train_count=len(train_records),
        test_count=len(test_records),
        category_count=model.n_categories_,
        metrics=metrics,
    )


def update_pipeline(
    model_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    save_features: bool = True,
) -> UpdateResult:
    """Incrementally update a saved ART-MAP model with newly labeled cases."""

    source_directory = Path(model_dir).resolve()
    destination = Path(output_dir).resolve()
    if source_directory == destination:
        raise ValueError(
            "Write incremental updates to a new output directory so the previous "
            "model remains reproducible."
        )
    model, config, parent_metadata = load_model_bundle(source_directory)
    records = load_manifest(manifest_path, require_labels=True)
    invalid_splits = {
        _normalized_split(record)
        for record in records
        if _normalized_split(record) not in {"", "train"}
    }
    if invalid_splits:
        raise ValueError(
            "Incremental-update manifests may use only blank or train split values; "
            f"found: {sorted(invalid_splits)}"
        )

    stratum = _experiment_stratum(records)
    _assert_patient_label_consistency(records)
    parent_run = parent_metadata.get("run", {})
    expected_stratum = parent_run.get("experiment_stratum")
    if stratum != expected_stratum:
        raise ValueError(
            "Incremental cases must match the saved plane, level, and sequence; "
            f"expected {expected_stratum}, found {stratum}."
        )

    provenance = parent_run.get("provenance", {})
    snapshot_name = provenance.get("snapshot_file")
    patient_hash_salt = provenance.get("patient_hash_salt")
    expected_snapshot_hash = provenance.get("snapshot_sha256")
    if not snapshot_name or not patient_hash_salt or not expected_snapshot_hash:
        raise ValueError(
            "The parent bundle lacks the deidentified provenance required for "
            "a leakage-safe incremental update."
        )
    parent_snapshot_path = source_directory / str(snapshot_name)
    if sha256_file(parent_snapshot_path) != expected_snapshot_hash:
        raise ValueError("The parent manifest snapshot failed its integrity check.")
    previous_snapshot_rows = load_manifest_snapshot(parent_snapshot_path)
    incremental_snapshot_rows = _snapshot_rows(
        records,
        partition="incremental_train",
        patient_hash_salt=str(patient_hash_salt),
    )

    previous_sample_ids = {row["sample_id"] for row in previous_snapshot_rows}
    repeated_sample_ids = sorted(
        row["sample_id"]
        for row in incremental_snapshot_rows
        if row["sample_id"] in previous_sample_ids
    )
    if repeated_sample_ids:
        raise ValueError(
            "Incremental sample IDs already exist in the model lineage: "
            + ", ".join(repeated_sample_ids)
        )
    held_out_patient_hashes = {
        row["patient_hash"]
        for row in previous_snapshot_rows
        if row["partition"] == "test"
    }
    leaked_samples = sorted(
        row["sample_id"]
        for row in incremental_snapshot_rows
        if row["patient_hash"] in held_out_patient_hashes
    )
    if leaked_samples:
        raise ValueError(
            "Incremental learning would include a held-out test patient; "
            "affected sample IDs: "
            + ", ".join(leaked_samples)
        )
    prior_training_labels: dict[str, set[str]] = defaultdict(set)
    for row in previous_snapshot_rows:
        if row["partition"] != "test":
            prior_training_labels[row["patient_hash"]].add(row["label"])
    conflicting_incremental_samples = sorted(
        row["sample_id"]
        for row in incremental_snapshot_rows
        if row["patient_hash"] in prior_training_labels
        and row["label"] not in prior_training_labels[row["patient_hash"]]
    )
    if conflicting_incremental_samples:
        raise ValueError(
            "Incremental labels conflict with the same training patient and stratum; "
            "affected sample IDs: "
            + ", ".join(conflicting_incremental_samples)
        )
    previous_image_hashes = {row["image_sha256"] for row in previous_snapshot_rows}
    new_image_hashes = [row["image_sha256"] for row in incremental_snapshot_rows]
    repeated_images = sorted(
        row["sample_id"]
        for row in incremental_snapshot_rows
        if row["image_sha256"] in previous_image_hashes
        or new_image_hashes.count(row["image_sha256"]) > 1
    )
    if repeated_images:
        raise ValueError(
            "Incremental images duplicate content already seen in this lineage: "
            + ", ".join(repeated_images)
        )

    destination = _prepare_output_directory(destination)

    extractor = ResNet18FeatureExtractor(
        pretrained=config.pretrained, device=config.device, seed=config.seed
    )
    feature_extractor_metadata = _extractor_metadata(extractor)
    _assert_extractor_matches(
        feature_extractor_metadata, parent_metadata.get("feature_extractor", {})
    )
    extraction = extractor.extract(
        [record.image_path for record in records], batch_size=config.batch_size
    )
    range_diagnostics, _ = _feature_range_diagnostics(model, extraction.features)
    out_of_range = int(range_diagnostics["out_of_training_range_value_count"])
    previous_category_count = model.n_categories_
    labels = [record.label for record in records]
    model.partial_fit(extraction.features, labels)

    if save_features:
        save_feature_matrix(
            destination / "incremental_features.npz",
            sample_ids=[record.sample_id for record in records],
            features=extraction.features,
        )
    combined_snapshot_rows = previous_snapshot_rows + incremental_snapshot_rows
    snapshot_path = save_manifest_snapshot(
        destination / "manifest_snapshot.csv", combined_snapshot_rows
    )
    run_metadata = {
        "run_type": "incremental_update",
        "parent_bundle": source_directory.name,
        "parent_created_at_utc": parent_metadata.get("created_at_utc"),
        "parent_metadata_sha256": sha256_file(source_directory / "metadata.json"),
        "manifest_name": Path(manifest_path).name,
        "experiment_stratum": stratum,
        "incremental_sample_ids": [record.sample_id for record in records],
        "incremental_sample_count": len(records),
        "incremental_patient_count": len({record.patient_id for record in records}),
        "previous_artmap_category_count": previous_category_count,
        "updated_artmap_category_count": model.n_categories_,
        "out_of_training_range_value_count": out_of_range,
        "feature_range_diagnostics": range_diagnostics,
        "scaler_bounds_remained_fixed": True,
        "training_class_counts_after_update": _distribution(
            [
                row["label"]
                for row in combined_snapshot_rows
                if row["partition"] != "test"
            ]
        ),
        "provenance": {
            "snapshot_file": snapshot_path.name,
            "snapshot_sha256": sha256_file(snapshot_path),
            "patient_hash_salt": patient_hash_salt,
            "raw_patient_ids_persisted": False,
            "raw_image_paths_persisted": False,
        },
    }
    save_model_bundle(
        destination,
        model=model,
        config=config,
        feature_extractor=feature_extractor_metadata,
        run_metadata=run_metadata,
    )
    return UpdateResult(
        output_dir=destination,
        sample_count=len(records),
        previous_category_count=previous_category_count,
        updated_category_count=model.n_categories_,
        out_of_training_range_value_count=out_of_range,
    )


def predict_images(
    model_dir: str | Path, image_paths: Sequence[str | Path]
) -> list[dict[str, Any]]:
    if not image_paths:
        raise ValueError("At least one image path is required.")
    model, config, metadata = load_model_bundle(model_dir)
    extractor = ResNet18FeatureExtractor(
        pretrained=config.pretrained, device=config.device, seed=config.seed
    )
    _assert_extractor_matches(
        _extractor_metadata(extractor), metadata.get("feature_extractor", {})
    )
    extraction = extractor.extract(image_paths, batch_size=config.batch_size)
    _, out_of_range_per_sample = _feature_range_diagnostics(
        model, extraction.features
    )
    predicted, details = model.predict_with_details(extraction.features)
    rows: list[dict[str, Any]] = []
    for path, label, detail, clipped_value_count in zip(
        extraction.paths,
        predicted,
        details,
        out_of_range_per_sample,
        strict=True,
    ):
        diagnostics = {key: value for key, value in detail.items() if key != "label"}
        rows.append(
            {
                "image_path": str(path),
                "predicted_label": None if label is None else str(label),
                "candidate_label": str(detail["label"]),
                **diagnostics,
                "out_of_training_range_value_count": int(clipped_value_count),
                "requires_review": bool(detail.get("rejected", False)),
                "research_use_only": True,
            }
        )
    return rows


def extract_manifest_features(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    pretrained: bool = True,
    device: str = "auto",
    batch_size: int = 8,
    seed: int = 42,
) -> Path:
    records = load_manifest(manifest_path, require_labels=False)
    extractor = ResNet18FeatureExtractor(
        pretrained=pretrained, device=device, seed=seed
    )
    result = extractor.extract(
        [record.image_path for record in records], batch_size=batch_size
    )
    return save_feature_matrix(
        output_path,
        sample_ids=[record.sample_id for record in records],
        features=result.features,
    )
