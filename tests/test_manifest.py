from __future__ import annotations

import csv
import tempfile
import unittest
import warnings
from pathlib import Path

from lumbar_stenosis_ai.data.manifest import (
    MANIFEST_COLUMNS,
    ManifestValidationError,
    PatientSplitLeakageWarning,
    load_manifest,
)


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "images").mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _touch_image(self, name: str) -> Path:
        path = self.root / "images" / name
        path.touch()
        return path

    def _write_manifest(
        self,
        rows: list[dict[str, str]],
        *,
        columns: tuple[str, ...] = MANIFEST_COLUMNS,
    ) -> Path:
        path = self.root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _row(**overrides: str) -> dict[str, str]:
        row = {
            "sample_id": "slice-001",
            "patient_id": "anon-001",
            "image_path": "images/slice-001.png",
            "label": "stenosis",
            "split": "train",
            "plane": "axial",
            "level": "L4-L5",
            "sequence": "T2",
        }
        row.update(overrides)
        return row

    def test_loads_schema_and_resolves_relative_image_path(self) -> None:
        image_path = self._touch_image("slice-001.png")
        manifest_path = self._write_manifest([self._row()])

        records = load_manifest(manifest_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_path, image_path.resolve())
        self.assertTrue(records[0].image_path.is_absolute())
        self.assertEqual(records[0].source_image_path, "images/slice-001.png")
        self.assertEqual(records[0].anonymized_patient_id, "anon-001")
        self.assertEqual(records[0].plane, "axial")
        self.assertEqual(records[0].level, "L4-L5")
        self.assertEqual(records[0].sequence, "T2")

    def test_sequence_header_is_required_but_value_may_be_blank(self) -> None:
        self._touch_image("slice-001.png")
        manifest_path = self._write_manifest([self._row(sequence="")])

        records = load_manifest(manifest_path)

        self.assertEqual(records[0].sequence, "")

        columns_without_sequence = tuple(
            column for column in MANIFEST_COLUMNS if column != "sequence"
        )
        row_without_sequence = self._row()
        row_without_sequence.pop("sequence")
        manifest_path = self._write_manifest(
            [row_without_sequence], columns=columns_without_sequence
        )
        with self.assertRaisesRegex(ManifestValidationError, "sequence"):
            load_manifest(manifest_path)

    def test_prediction_manifest_may_omit_label(self) -> None:
        self._touch_image("slice-001.png")
        manifest_path = self._write_manifest([self._row(label="")])

        records = load_manifest(manifest_path, require_labels=False)

        self.assertEqual(records[0].label, "")
        with self.assertRaisesRegex(ManifestValidationError, "empty required values: label"):
            load_manifest(manifest_path)

    def test_blank_split_is_allowed_for_automatic_patient_split(self) -> None:
        self._touch_image("slice-001.png")
        manifest_path = self._write_manifest([self._row(split="")])

        records = load_manifest(manifest_path)

        self.assertEqual(records[0].split, "")

    def test_rejects_duplicate_sample_and_image_path(self) -> None:
        self._touch_image("slice-001.png")
        rows = [
            self._row(),
            self._row(split="test"),
        ]
        manifest_path = self._write_manifest(rows)

        with self.assertRaises(ManifestValidationError) as context:
            load_manifest(manifest_path, patient_split_policy="ignore")

        self.assertIn("duplicate sample_id", str(context.exception))
        self.assertIn("duplicate image_path", str(context.exception))

    def test_rejects_missing_image_and_invalid_split(self) -> None:
        manifest_path = self._write_manifest([self._row(split="development")])

        with self.assertRaises(ManifestValidationError) as context:
            load_manifest(manifest_path)

        self.assertIn("unsupported split", str(context.exception))
        self.assertIn("does not exist", str(context.exception))

    def test_patient_in_multiple_splits_is_error_by_default(self) -> None:
        self._touch_image("slice-001.png")
        self._touch_image("slice-002.png")
        rows = [
            self._row(),
            self._row(
                sample_id="slice-002",
                image_path="images/slice-002.png",
                split="test",
            ),
        ]
        manifest_path = self._write_manifest(rows)

        with self.assertRaisesRegex(
            ManifestValidationError, "patient-level data leakage"
        ):
            load_manifest(manifest_path)

    def test_patient_in_multiple_splits_can_emit_warning(self) -> None:
        self._touch_image("slice-001.png")
        self._touch_image("slice-002.png")
        rows = [
            self._row(),
            self._row(
                sample_id="slice-002",
                image_path="images/slice-002.png",
                split="test",
            ),
        ]
        manifest_path = self._write_manifest(rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = load_manifest(manifest_path, patient_split_policy="warn")

        self.assertEqual(len(records), 2)
        self.assertTrue(
            any(item.category is PatientSplitLeakageWarning for item in caught)
        )

    def test_rejects_phi_column_explicitly(self) -> None:
        columns = MANIFEST_COLUMNS + ("patient_name",)
        row = self._row()
        row["patient_name"] = "Not Allowed"
        manifest_path = self._write_manifest([row], columns=columns)

        with self.assertRaisesRegex(ManifestValidationError, "PHI"):
            load_manifest(manifest_path, check_files=False)


if __name__ == "__main__":
    unittest.main()
