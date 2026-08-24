import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lumbar_stenosis_ai.config import PipelineConfig
from lumbar_stenosis_ai.data import ManifestRecord
from lumbar_stenosis_ai.features.resnet18 import ExtractionResult
from lumbar_stenosis_ai.pipeline import (
    split_manifest_records,
    train_pipeline,
    update_pipeline,
)


def record(sample_id, patient_id, label, split=""):
    return ManifestRecord(
        sample_id=sample_id,
        patient_id=patient_id,
        image_path=Path(f"{sample_id}.png"),
        label=label,
        split=split,
        plane="axial",
        level="L4-L5",
        sequence="T2",
    )


class PipelineSplitTests(unittest.TestCase):
    def test_explicit_split(self):
        records = [
            record("s1", "p1", "a", "train"),
            record("s2", "p2", "a", "test"),
        ]
        train, test, method = split_manifest_records(records, PipelineConfig())
        self.assertEqual([item.sample_id for item in train], ["s1"])
        self.assertEqual([item.sample_id for item in test], ["s2"])
        self.assertEqual(method, "explicit_patient_split")

    def test_mixed_explicit_and_automatic_split_rejected(self):
        records = [
            record("s1", "p1", "a", "train"),
            record("s2", "p2", "a", ""),
        ]
        with self.assertRaises(ValueError):
            split_manifest_records(records, PipelineConfig())

    def test_training_pipeline_writes_unseen_test_metrics(self):
        class FakeExtractor:
            def __init__(self, **_):
                self._metadata = {
                    "name": "ResNet18",
                    "output_features": 512,
                    "pretrained_imagenet": False,
                    "initialization_seed": 42,
                    "weights_id": "fake_weights",
                    "preprocessing_id": "fake_preprocessing",
                    "state_sha256": "f" * 64,
                }

            def metadata(self):
                return dict(self._metadata)

            def extract(self, paths, *, batch_size):
                del batch_size
                resolved = tuple(Path(path).resolve() for path in paths)
                feature_map = {
                    "a_train.img": [0.0, 0.1],
                    "b_train.img": [0.9, 1.0],
                    "a_test.img": [0.05, 0.05],
                    "b_test.img": [0.95, 0.95],
                    "c_new.img": [0.5, 0.5],
                }
                return ExtractionResult(
                    paths=resolved,
                    features=np.asarray([feature_map[path.name] for path in resolved]),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(
                ("a_train.img", "b_train.img", "a_test.img", "b_test.img"),
                start=1,
            ):
                (root / name).write_bytes(bytes([index]))
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,patient_id,image_path,label,split,plane,level,sequence\n"
                "a1,p1,a_train.img,a,train,axial,L4-L5,T2\n"
                "b1,p2,b_train.img,b,train,axial,L4-L5,T2\n"
                "a2,p3,a_test.img,a,test,axial,L4-L5,T2\n"
                "b2,p4,b_test.img,b,test,axial,L4-L5,T2\n",
                encoding="utf-8",
            )
            output_dir = root / "run"
            with patch("lumbar_stenosis_ai.pipeline.ResNet18FeatureExtractor", FakeExtractor):
                result = train_pipeline(
                    manifest,
                    output_dir,
                    config=PipelineConfig(pretrained=False),
                )
            self.assertEqual(result.test_count, 2)
            self.assertEqual(result.metrics["accuracy"], 1.0)
            self.assertEqual(result.metrics["patient_level"]["accuracy"], 1.0)
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "test_predictions.csv").is_file())
            self.assertTrue((output_dir / "patient_test_predictions.csv").is_file())
            self.assertTrue((output_dir / "metadata.json").is_file())

            (root / "c_new.img").write_bytes(b"new-case")
            update_manifest = root / "update.csv"
            update_manifest.write_text(
                "sample_id,patient_id,image_path,label,split,plane,level,sequence\n"
                "c1,p5,c_new.img,c,,axial,L4-L5,T2\n",
                encoding="utf-8",
            )
            updated_dir = root / "updated_run"
            with patch("lumbar_stenosis_ai.pipeline.ResNet18FeatureExtractor", FakeExtractor):
                update_result = update_pipeline(output_dir, update_manifest, updated_dir)
            self.assertEqual(update_result.sample_count, 1)
            self.assertGreaterEqual(
                update_result.updated_category_count,
                update_result.previous_category_count,
            )
            self.assertTrue((updated_dir / "metadata.json").is_file())
            self.assertTrue((updated_dir / "manifest_snapshot.csv").is_file())

            # A patient reserved for the original unseen test set must never
            # be learned later under a different sample ID.
            (root / "held_out_new.img").write_bytes(b"held-out-new")
            leakage_manifest = root / "leakage.csv"
            leakage_manifest.write_text(
                "sample_id,patient_id,image_path,label,split,plane,level,sequence\n"
                "d1,p3,held_out_new.img,a,,axial,L4-L5,T2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "held-out test patient"):
                update_pipeline(output_dir, leakage_manifest, root / "forbidden_update")

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            (output / "stale.txt").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "new or empty"):
                # Validation reaches the output-integrity guard only after a
                # real manifest, so exercise the small guard directly.
                from lumbar_stenosis_ai.pipeline import _prepare_output_directory

                _prepare_output_directory(output)


if __name__ == "__main__":
    unittest.main()
