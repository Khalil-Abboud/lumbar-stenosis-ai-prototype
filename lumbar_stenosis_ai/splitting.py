"""Patient-level splitting to prevent slice leakage between train and test."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import TypeVar


RecordT = TypeVar("RecordT")


def patient_level_split(
    records: Sequence[RecordT],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[RecordT], list[RecordT]]:
    """Create a deterministic, approximately stratified patient holdout.

    Records must expose ``patient_id`` and ``label`` attributes.  All slices
    belonging to one patient remain in the same partition.
    """

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1) for automatic splitting.")
    if not records:
        raise ValueError("Cannot split an empty dataset.")

    by_patient: dict[str, list[RecordT]] = defaultdict(list)
    for record in records:
        patient_id = str(getattr(record, "patient_id"))
        label = str(getattr(record, "label"))
        if not patient_id:
            raise ValueError("Every record needs an anonymized patient_id.")
        if not label:
            raise ValueError("Every training record needs a label.")
        by_patient[patient_id].append(record)

    if len(by_patient) < 2:
        raise ValueError("At least two patients are required for a patient-level split.")

    patient_label: dict[str, str] = {}
    for patient_id, patient_records in by_patient.items():
        counts = Counter(str(getattr(record, "label")) for record in patient_records)
        patient_label[patient_id] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    buckets: dict[str, list[str]] = defaultdict(list)
    for patient_id, label in patient_label.items():
        buckets[label].append(patient_id)

    rng = random.Random(seed)
    test_patients: set[str] = set()
    for label in sorted(buckets):
        patient_ids = sorted(buckets[label])
        rng.shuffle(patient_ids)
        if len(patient_ids) == 1:
            continue
        holdout_count = max(1, round(len(patient_ids) * test_fraction))
        holdout_count = min(holdout_count, len(patient_ids) - 1)
        test_patients.update(patient_ids[:holdout_count])

    if not test_patients:
        raise ValueError(
            "A leakage-safe test set could not be created. Provide at least two "
            "patients in one or more classes, or define explicit train/test splits."
        )

    train = [record for record in records if str(getattr(record, "patient_id")) not in test_patients]
    test = [record for record in records if str(getattr(record, "patient_id")) in test_patients]
    if not train or not test:
        raise RuntimeError("Patient-level splitting produced an empty partition.")
    return train, test
