import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from lumbar_stenosis_ai.data import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    load_manifest,
    prepare_multidisorder_manifest,
)


class MultiDisorderAdapterTests(unittest.TestCase):
    def _image(self, path: Path, value: int = 64) -> None:
        Image.new("L", (8, 8), color=value).save(path)

    def test_builds_patient_consistent_t2_manifest_without_mask_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            images = root / "images"
            stenosis = root / "labeled_images" / "Spinal Stenosis (SS)"
            images.mkdir(parents=True)
            stenosis.mkdir(parents=True)

            self._image(images / "Sagittal_0001_T2.png")
            self._image(images / "Sagittal_0001_T2_1.png")
            self._image(images / "Sagittal_0002_T2.png", value=65)
            self._image(images / "Sagittal_0003_T1.png")
            self._image(stenosis / "Sagittal_0001_T2_mask.png", value=1)

            output = Path(directory) / "manifest.csv"
            result = prepare_multidisorder_manifest(root, output)
            records = load_manifest(output)

            self.assertEqual(result.sample_count, 2)
            self.assertEqual(result.patient_count, 2)
            self.assertEqual(result.duplicate_images_removed, 1)
            self.assertEqual(
                [record.sequence for record in records], ["T2", "T2"]
            )
            labels = {record.sample_id: record.label for record in records}
            self.assertEqual(labels["Sagittal_0001_T2"], POSITIVE_LABEL)
            self.assertEqual(labels["Sagittal_0002_T2"], NEGATIVE_LABEL)
            self.assertTrue(all("labeled_images" not in str(r.image_path) for r in records))
            self.assertEqual({record.level for record in records}, {"whole-lumbar"})

            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row["split"] == "" for row in rows))

    def test_rejects_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            images = root / "images"
            stenosis = root / "labeled_images" / "Spinal Stenosis (SS)"
            images.mkdir(parents=True)
            stenosis.mkdir(parents=True)
            self._image(images / "Sagittal_0001_T2.png")
            self._image(images / "Sagittal_0002_T2.png", value=65)
            self._image(stenosis / "Sagittal_0001_T2_mask.png", value=1)
            output = Path(directory) / "manifest.csv"
            prepare_multidisorder_manifest(root, output)

            with self.assertRaises(FileExistsError):
                prepare_multidisorder_manifest(root, output)


if __name__ == "__main__":
    unittest.main()
