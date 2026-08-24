"""CSV manifest loading and validation for anonymized imaging data.

The manifest deliberately has a small, fixed schema.  Keeping personally
identifying fields out of this file makes it safer to use in experiments and
to share alongside derived results.
"""

from __future__ import annotations

import csv
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


MANIFEST_COLUMNS = (
    "sample_id",
    "patient_id",
    "image_path",
    "label",
    "split",
    "plane",
    "level",
    "sequence",
)

ALLOWED_SPLITS = frozenset({"train", "test"})

# These are checked before the general fixed-schema check so that an unsafe
# manifest produces an explicit privacy-oriented error message.
_PHI_COLUMN_NAMES = frozenset(
    {
        "name",
        "patientname",
        "firstname",
        "middlename",
        "lastname",
        "fullname",
        "surname",
        "birthdate",
        "dateofbirth",
        "dob",
        "address",
        "email",
        "phone",
        "telephone",
        "passport",
        "snils",
        "medicalrecordnumber",
        "mrn",
        "insuranceid",
    }
)

PatientSplitPolicy = Literal["error", "warn", "ignore"]


class ManifestValidationError(ValueError):
    """Raised when a dataset manifest is unsafe or internally inconsistent."""

    def __init__(self, issues: str | Iterable[str]):
        if isinstance(issues, str):
            self.issues = (issues,)
        else:
            self.issues = tuple(issues)

        message = "Manifest validation failed:\n- " + "\n- ".join(self.issues)
        super().__init__(message)


class PatientSplitLeakageWarning(UserWarning):
    """Warns that slices from one patient occur in multiple data splits."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One anonymized image sample from a manifest.

    ``image_path`` is always absolute after :func:`load_manifest` returns.
    ``source_image_path`` preserves the value written in the CSV, which is
    useful when reporting a malformed row without losing portability.
    """

    sample_id: str
    patient_id: str
    image_path: Path
    label: str
    split: str
    plane: str = ""
    level: str = ""
    source_image_path: str = ""
    sequence: str = ""

    @property
    def anonymized_patient_id(self) -> str:
        """An explicit alias documenting the intended meaning of patient_id."""

        return self.patient_id


def _normalized_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _validate_header(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ManifestValidationError("the CSV is empty or has no header")

    if any(not name for name in fieldnames):
        raise ManifestValidationError("the CSV header contains an empty column name")

    duplicate_headers = sorted(
        {name for name in fieldnames if fieldnames.count(name) > 1}
    )
    if duplicate_headers:
        raise ManifestValidationError(
            f"duplicate CSV columns: {', '.join(duplicate_headers)}"
        )

    phi_columns = sorted(
        name
        for name in fieldnames
        if _normalized_column_name(name) in _PHI_COLUMN_NAMES
    )
    if phi_columns:
        raise ManifestValidationError(
            "personally identifying (PHI) columns are not permitted: "
            + ", ".join(phi_columns)
        )

    expected = set(MANIFEST_COLUMNS)
    actual = set(fieldnames)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    issues: list[str] = []
    if missing:
        issues.append("missing required columns: " + ", ".join(missing))
    if extra:
        issues.append(
            "unsupported columns: "
            + ", ".join(extra)
            + "; use only the anonymized manifest schema"
        )
    if issues:
        raise ManifestValidationError(issues)


def _absolute_image_path(raw_path: str, manifest_directory: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_directory / candidate
    return candidate.resolve(strict=False)


def load_manifest(
    path: str | Path,
    *,
    require_labels: bool = True,
    check_files: bool = True,
    patient_split_policy: PatientSplitPolicy = "error",
    allowed_splits: Iterable[str] = ALLOWED_SPLITS,
) -> list[ManifestRecord]:
    """Load and validate a UTF-8 CSV dataset manifest.

    Relative image paths are resolved against the manifest's directory, not
    the process working directory.  Set ``require_labels=False`` for an
    inference-only manifest.  Patient-level split validation defaults to an
    error because slice-level splitting can otherwise leak nearly identical
    images into both training and evaluation.
    """

    manifest_path = Path(path).expanduser().resolve(strict=False)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_path}")

    records: list[ManifestRecord] = []
    row_issues: list[str] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is not None:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        _validate_header(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            values = {
                column: (row.get(column) or "").strip()
                for column in MANIFEST_COLUMNS
            }

            # Silently ignore wholly blank lines, which spreadsheet programs
            # commonly append to otherwise valid CSV files.
            if not any(values.values()):
                continue

            missing_values = [
                column
                for column in ("sample_id", "patient_id", "image_path")
                if not values[column]
            ]
            if require_labels and not values["label"]:
                missing_values.append("label")
            if missing_values:
                row_issues.append(
                    f"line {line_number} has empty required values: "
                    + ", ".join(missing_values)
                )
                continue

            source_image_path = values["image_path"]
            records.append(
                ManifestRecord(
                    sample_id=values["sample_id"],
                    patient_id=values["patient_id"],
                    image_path=_absolute_image_path(
                        source_image_path, manifest_path.parent
                    ),
                    label=values["label"],
                    split=values["split"].casefold(),
                    plane=values["plane"],
                    level=values["level"],
                    sequence=values["sequence"],
                    source_image_path=source_image_path,
                )
            )

    if row_issues:
        raise ManifestValidationError(row_issues)

    validate_manifest(
        records,
        require_labels=require_labels,
        check_files=check_files,
        patient_split_policy=patient_split_policy,
        allowed_splits=allowed_splits,
    )
    return records


def validate_manifest(
    records: Iterable[ManifestRecord],
    *,
    require_labels: bool = True,
    check_files: bool = True,
    patient_split_policy: PatientSplitPolicy = "error",
    allowed_splits: Iterable[str] = ALLOWED_SPLITS,
) -> None:
    """Validate manifest records, raising one error with all detected issues."""

    if patient_split_policy not in {"error", "warn", "ignore"}:
        raise ValueError(
            "patient_split_policy must be one of: 'error', 'warn', 'ignore'"
        )

    materialized_records = list(records)
    if not materialized_records:
        raise ManifestValidationError("the manifest contains no sample rows")

    normalized_allowed_splits = {split.casefold() for split in allowed_splits}
    issues: list[str] = []
    seen_sample_ids: dict[str, int] = {}
    seen_image_paths: dict[str, str] = {}
    patient_splits: dict[str, set[str]] = defaultdict(set)

    for index, record in enumerate(materialized_records, start=1):
        descriptor = f"sample {record.sample_id!r}" if record.sample_id else f"row {index}"

        if not record.sample_id:
            issues.append(f"row {index} has an empty sample_id")
        elif record.sample_id in seen_sample_ids:
            issues.append(
                f"duplicate sample_id {record.sample_id!r} "
                f"(rows {seen_sample_ids[record.sample_id]} and {index})"
            )
        else:
            seen_sample_ids[record.sample_id] = index

        if not record.patient_id:
            issues.append(f"{descriptor} has an empty anonymized patient_id")

        split = record.split.casefold()
        # A blank split asks the training pipeline to create a deterministic
        # patient-level split.  Nonblank values must use the declared schema.
        if split and split not in normalized_allowed_splits:
            issues.append(
                f"{descriptor} has unsupported split {record.split!r}; allowed values: "
                + ", ".join(sorted(normalized_allowed_splits))
            )

        if require_labels and not record.label.strip():
            issues.append(f"{descriptor} has an empty label")

        normalized_path = os.path.normcase(os.fspath(record.image_path.resolve(strict=False)))
        previous_sample = seen_image_paths.get(normalized_path)
        if previous_sample is not None:
            issues.append(
                f"duplicate image_path used by samples {previous_sample!r} "
                f"and {record.sample_id!r}: {record.image_path}"
            )
        else:
            seen_image_paths[normalized_path] = record.sample_id

        if check_files and not record.image_path.is_file():
            issues.append(f"image file for {descriptor} does not exist: {record.image_path}")

        if record.patient_id and split:
            patient_splits[record.patient_id].add(split)

    overlapping_patients = {
        patient_id: splits
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    }
    if overlapping_patients and patient_split_policy != "ignore":
        details = "; ".join(
            f"{patient_id!r}: {', '.join(sorted(splits))}"
            for patient_id, splits in sorted(overlapping_patients.items())
        )
        leakage_message = (
            "patients occur in multiple splits (patient-level data leakage): " + details
        )
        if patient_split_policy == "error":
            issues.append(leakage_message)
        else:
            warnings.warn(leakage_message, PatientSplitLeakageWarning, stacklevel=2)

    if issues:
        raise ManifestValidationError(issues)


# A discoverable alias for callers that naturally look for ``read_manifest``.
read_manifest = load_manifest


__all__ = [
    "ALLOWED_SPLITS",
    "MANIFEST_COLUMNS",
    "ManifestRecord",
    "ManifestValidationError",
    "PatientSplitLeakageWarning",
    "load_manifest",
    "read_manifest",
    "validate_manifest",
]
