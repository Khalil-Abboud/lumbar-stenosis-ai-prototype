"""Adapter for the public Multi-Disorder sagittal lumbar MRI dataset.

The dataset stores raw sagittal PNG images in ``images`` and one binary mask
per annotated disorder in class-specific folders below ``labeled_images``.
This adapter uses the *presence* of a non-empty spinal-stenosis mask to build
an image-classification manifest.  Masks are never used as model inputs.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .manifest import MANIFEST_COLUMNS


STENOSIS_MASK_DIRECTORY = "Spinal Stenosis (SS)"
POSITIVE_LABEL = "stenosis_present"
NEGATIVE_LABEL = "stenosis_annotation_absent"
_IMAGE_NAME = re.compile(
    r"^Sagittal_(?P<patient>\d{4})_(?P<sequence>T1|T2)(?:_\d+)?\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MultiDisorderPreparationResult:
    """Summary of one generated pilot manifest."""

    manifest_path: Path
    sequence: str
    sample_count: int
    patient_count: int
    sample_labels: dict[str, int]
    patient_labels: dict[str, int]
    empty_stenosis_masks: int
    duplicate_images_removed: int
    selected_patient_limit_per_class: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "completed",
            "manifest": str(self.manifest_path),
            "sequence": self.sequence,
            "sample_count": self.sample_count,
            "patient_count": self.patient_count,
            "sample_labels": self.sample_labels,
            "patient_labels": self.patient_labels,
            "empty_stenosis_masks_ignored": self.empty_stenosis_masks,
            "duplicate_images_removed": self.duplicate_images_removed,
            "selected_patient_limit_per_class": self.selected_patient_limit_per_class,
            "label_definition": {
                POSITIVE_LABEL: "a non-empty Spinal Stenosis mask exists",
                NEGATIVE_LABEL: "no Spinal Stenosis mask exists for this patient group",
            },
            "masks_used_as_model_input": False,
            "pilot_only": True,
        }


def _dataset_directories(dataset_root: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_root).expanduser().resolve(strict=False)
    images = root / "images"
    masks = root / "labeled_images" / STENOSIS_MASK_DIRECTORY
    missing = [str(path) for path in (images, masks) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "The Multi-Disorder dataset structure is incomplete; missing: "
            + ", ".join(missing)
        )
    return images, masks


def _nonempty_mask(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            # The published masks are 16-bit PNG files (Pillow mode I;16).
            # Pillow's getbbox() can report None for non-zero I;16 images, so
            # inspect the decoded samples directly instead.
            return bool(np.any(np.asarray(image) != 0))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read annotation mask {path}: {exc}") from exc


def _source_name_from_mask(mask_path: Path) -> str:
    suffix = "_mask.png"
    if not mask_path.name.casefold().endswith(suffix):
        raise ValueError(f"Unexpected stenosis-mask filename: {mask_path.name}")
    return mask_path.name[: -len(suffix)] + ".png"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_patients(
    labels_by_patient: dict[str, str],
    *,
    maximum_per_class: int | None,
    seed: int,
) -> set[str]:
    if maximum_per_class is None:
        return set(labels_by_patient)
    if maximum_per_class < 2:
        raise ValueError("max_patients_per_class must be at least 2")

    by_label: dict[str, list[str]] = defaultdict(list)
    for patient_id, label in labels_by_patient.items():
        by_label[label].append(patient_id)

    rng = random.Random(seed)
    selected: set[str] = set()
    for label in sorted(by_label):
        patient_ids = sorted(by_label[label])
        rng.shuffle(patient_ids)
        selected.update(patient_ids[:maximum_per_class])
    return selected


def prepare_multidisorder_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    sequence: str = "T2",
    max_patients_per_class: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> MultiDisorderPreparationResult:
    """Create a leakage-safe binary manifest from the public PNG dataset.

    A source identifier can have more than one image variant.  All variants
    are grouped under one anonymized patient ID and receive a positive label
    if any variant has a non-empty stenosis annotation.  This prevents a
    duplicate/variant image from crossing the patient-level train/test split.

    The negative label means only that this dataset contains no stenosis mask
    for the patient group.  It must not be interpreted as a clinical finding.
    """

    normalized_sequence = sequence.strip().upper()
    if normalized_sequence not in {"T1", "T2"}:
        raise ValueError("sequence must be T1 or T2")

    images_directory, masks_directory = _dataset_directories(dataset_root)
    output = Path(output_path).expanduser().resolve(strict=False)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {output}. Use --overwrite to replace it."
        )

    image_rows: list[tuple[Path, str, str]] = []
    source_names: set[str] = set()
    for image_path in sorted(images_directory.glob("*.png")):
        match = _IMAGE_NAME.fullmatch(image_path.name)
        if match is None or match.group("sequence").upper() != normalized_sequence:
            continue
        patient_id = f"md_{match.group('patient')}"
        image_rows.append((image_path.resolve(), patient_id, image_path.stem))
        source_names.add(image_path.name.casefold())

    if not image_rows:
        raise ValueError(
            f"No {normalized_sequence} sagittal PNG images were found in "
            f"{images_directory}"
        )

    positive_source_names: set[str] = set()
    empty_masks = 0
    orphan_masks: list[str] = []
    for mask_path in sorted(masks_directory.glob("*.png")):
        source_name = _source_name_from_mask(mask_path)
        match = _IMAGE_NAME.fullmatch(source_name)
        if match is None or match.group("sequence").upper() != normalized_sequence:
            continue
        if source_name.casefold() not in source_names:
            orphan_masks.append(mask_path.name)
            continue
        if _nonempty_mask(mask_path):
            positive_source_names.add(source_name.casefold())
        else:
            empty_masks += 1

    if orphan_masks:
        preview = ", ".join(orphan_masks[:5])
        raise ValueError(
            "Stenosis masks do not have matching source images: "
            + preview
            + (" ..." if len(orphan_masks) > 5 else "")
        )
    if not positive_source_names:
        raise ValueError(
            f"No non-empty {normalized_sequence} spinal-stenosis masks were found."
        )

    positive_patients = {
        patient_id
        for image_path, patient_id, _ in image_rows
        if image_path.name.casefold() in positive_source_names
    }
    labels_by_patient = {
        patient_id: (
            POSITIVE_LABEL if patient_id in positive_patients else NEGATIVE_LABEL
        )
        for _, patient_id, _ in image_rows
    }
    if len(set(labels_by_patient.values())) != 2:
        raise ValueError("Both positive and annotation-absent patient groups are required.")

    # The published collection contains a few byte-identical image variants.
    # Keep one deterministic copy per patient so duplicate content cannot
    # inflate training or appear twice in provenance checks.
    deduplicated_rows: list[tuple[Path, str, str]] = []
    content_owner: dict[str, str] = {}
    duplicate_images_removed = 0
    for image_path, patient_id, sample_id in image_rows:
        image_hash = _sha256_file(image_path)
        previous_patient = content_owner.get(image_hash)
        if previous_patient is None:
            content_owner[image_hash] = patient_id
            deduplicated_rows.append((image_path, patient_id, sample_id))
            continue
        if previous_patient != patient_id:
            raise ValueError(
                "Exact duplicate image content occurs under different patient IDs: "
                f"{previous_patient} and {patient_id}"
            )
        duplicate_images_removed += 1

    retained_patients = {patient_id for _, patient_id, _ in deduplicated_rows}
    labels_by_patient = {
        patient_id: label
        for patient_id, label in labels_by_patient.items()
        if patient_id in retained_patients
    }
    selected_patients = _select_patients(
        labels_by_patient,
        maximum_per_class=max_patients_per_class,
        seed=seed,
    )

    rows: list[dict[str, str]] = []
    for image_path, patient_id, sample_id in deduplicated_rows:
        if patient_id not in selected_patients:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "image_path": str(image_path),
                "label": labels_by_patient[patient_id],
                "split": "",
                "plane": "sagittal",
                "level": "whole-lumbar",
                "sequence": normalized_sequence,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    sample_labels = Counter(row["label"] for row in rows)
    patient_labels = Counter(
        labels_by_patient[patient_id] for patient_id in selected_patients
    )
    return MultiDisorderPreparationResult(
        manifest_path=output,
        sequence=normalized_sequence,
        sample_count=len(rows),
        patient_count=len(selected_patients),
        sample_labels=dict(sorted(sample_labels.items())),
        patient_labels=dict(sorted(patient_labels.items())),
        empty_stenosis_masks=empty_masks,
        duplicate_images_removed=duplicate_images_removed,
        selected_patient_limit_per_class=max_patients_per_class,
    )


__all__ = [
    "NEGATIVE_LABEL",
    "POSITIVE_LABEL",
    "STENOSIS_MASK_DIRECTORY",
    "MultiDisorderPreparationResult",
    "prepare_multidisorder_manifest",
]
