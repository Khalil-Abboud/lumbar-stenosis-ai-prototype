import tempfile
import unittest
from pathlib import Path

import numpy as np

from lumbar_stenosis_ai.artifacts import (
    load_model_bundle,
    save_feature_matrix,
    save_model_bundle,
)
from lumbar_stenosis_ai.config import PipelineConfig
from lumbar_stenosis_ai.models import FuzzyARTMAPClassifier


class ArtifactTests(unittest.TestCase):
    def test_model_round_trip_preserves_predictions(self):
        x = np.asarray([[0.0, 0.1], [0.1, 0.0], [0.9, 1.0], [1.0, 0.9]])
        y = ["a", "a", "b", "b"]
        model = FuzzyARTMAPClassifier(vigilance=0.7).fit(x, y)
        expected = model.predict([[0.05, 0.05], [0.95, 0.95]]).tolist()

        with tempfile.TemporaryDirectory() as directory:
            save_model_bundle(
                directory,
                model=model,
                config=PipelineConfig(pretrained=False),
                feature_extractor={
                    "name": "ResNet18",
                    "output_features": 512,
                    "weights_id": "test_weights",
                    "preprocessing_id": "test_preprocessing",
                    "state_sha256": "0" * 64,
                },
                run_metadata={"test": True},
            )
            restored, config, metadata = load_model_bundle(directory)
            actual = restored.predict([[0.05, 0.05], [0.95, 0.95]]).tolist()
            self.assertEqual(actual, expected)
            self.assertFalse(config.pretrained)
            self.assertTrue(metadata["research_use_only"])
            self.assertTrue((Path(directory) / "model_state.npz").is_file())

    def test_feature_matrix_reports_actual_npz_path(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "features"
            actual = save_feature_matrix(
                requested,
                sample_ids=["s1"],
                features=np.asarray([[1.0, 2.0]]),
            )
            self.assertEqual(actual, Path(f"{requested}.npz"))
            self.assertTrue(actual.is_file())
            self.assertFalse(requested.exists())


if __name__ == "__main__":
    unittest.main()
