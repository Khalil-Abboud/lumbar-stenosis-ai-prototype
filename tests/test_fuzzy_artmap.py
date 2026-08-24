"""Focused tests for the dependency-light Fuzzy ARTMAP classifier."""

import json
import unittest

import numpy as np

from lumbar_stenosis_ai.models import FuzzyARTMAPClassifier, MinMaxFeatureScaler


class MinMaxFeatureScalerTests(unittest.TestCase):
    def test_scales_and_clips_with_constant_feature(self) -> None:
        scaler = MinMaxFeatureScaler().fit([[0.0, 5.0], [10.0, 5.0]])

        transformed = scaler.transform([[-5.0, 5.0], [15.0, 5.0]])

        np.testing.assert_allclose(transformed, [[0.0, 0.0], [1.0, 0.0]])

    def test_scaler_state_round_trip(self) -> None:
        original = MinMaxFeatureScaler().fit([[1.0, 3.0], [5.0, 9.0]])
        restored = MinMaxFeatureScaler.from_dict(original.to_dict())

        np.testing.assert_allclose(
            restored.transform([[3.0, 6.0]]), original.transform([[3.0, 6.0]])
        )

    def test_unclipped_scaler_rejects_change_in_constant_feature(self) -> None:
        scaler = MinMaxFeatureScaler(clip=False).fit([[1.0, 5.0], [2.0, 5.0]])

        with self.assertRaisesRegex(ValueError, "outside the fitted feature range"):
            scaler.transform([[1.5, 6.0]])


class FuzzyARTMAPClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.asarray(
            [
                [0.05, 0.10],
                [0.10, 0.05],
                [0.45, 0.50],
                [0.50, 0.45],
                [0.90, 0.95],
                [0.95, 0.90],
            ]
        )
        self.y = np.asarray(
            ["normal", "normal", "moderate", "moderate", "severe", "severe"]
        )

    def test_complement_coding_preserves_constant_l1_norm(self) -> None:
        encoded = FuzzyARTMAPClassifier.complement_code(
            [[0.0, 0.25], [0.75, 1.0]]
        )

        self.assertEqual(encoded.shape, (2, 4))
        np.testing.assert_allclose(encoded.sum(axis=1), [2.0, 2.0])

    def test_fit_predicts_separable_string_classes(self) -> None:
        model = FuzzyARTMAPClassifier(vigilance=0.8).fit(self.X, self.y)

        predictions = model.predict(self.X)

        np.testing.assert_array_equal(predictions, self.y)
        np.testing.assert_array_equal(
            model.classes_, ["normal", "moderate", "severe"]
        )
        self.assertGreaterEqual(model.n_categories_, 3)

    def test_label_mismatch_uses_match_tracking_and_commits_new_category(self) -> None:
        # Identical feature locations with different supervised labels cannot
        # share a category; match tracking must force a second category.
        model = FuzzyARTMAPClassifier(vigilance=0.2).fit(
            [[0.2], [0.2]], ["class-a", "class-b"]
        )

        self.assertEqual(model.n_categories_, 2)
        self.assertEqual(model.category_labels_, ["class-a", "class-b"])

    def test_partial_fit_keeps_scaler_fixed_and_adds_new_class(self) -> None:
        model = FuzzyARTMAPClassifier(vigilance=0.8).fit(
            self.X[:4], self.y[:4]
        )
        bounds_before = model.scaler_.to_dict()

        model.partial_fit(self.X[4:], self.y[4:])

        self.assertEqual(model.scaler_.to_dict(), bounds_before)
        self.assertIn("severe", model.classes_.tolist())
        np.testing.assert_array_equal(model.predict(self.X[4:]), self.y[4:])

    def test_prediction_details_report_choice_match_and_vigilance(self) -> None:
        model = FuzzyARTMAPClassifier(vigilance=0.7).fit(self.X, self.y)

        predictions, details = model.predict_with_details([[0.08, 0.08]])

        self.assertEqual(predictions.tolist(), ["normal"])
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["label"], "normal")
        self.assertIsInstance(details[0]["category"], int)
        self.assertGreater(details[0]["choice"], 0.0)
        self.assertGreaterEqual(details[0]["match"], 0.0)
        self.assertLessEqual(details[0]["match"], 1.0)
        self.assertTrue(details[0]["meets_vigilance"])
        self.assertFalse(details[0]["rejected"])

    def test_prediction_searches_past_top_choice_for_resonating_category(self) -> None:
        # For input [0.5], category 0 has the highest choice because it is very
        # specific, but its match is only 0.6.  Category 1 has lower choice and
        # match 0.8, so it is the first category satisfying vigilance 0.75.
        model = FuzzyARTMAPClassifier.from_dict(
            {
                "model": "FuzzyARTMAPClassifier",
                "version": 1,
                "parameters": {
                    "vigilance": 0.75,
                    "alpha": 0.001,
                    "beta": 1.0,
                    "match_tracking_epsilon": 1e-6,
                    "clip": True,
                },
                "scaler": {
                    "clip": True,
                    "feature_min": [0.0],
                    "feature_max": [1.0],
                },
                "state": {
                    "n_features_in": 1,
                    "weights": [[0.5, 0.1], [0.3, 0.7]],
                    "category_labels": ["top-choice", "resonating"],
                    "category_counts": [2, 1],
                },
            }
        )

        predictions, details = model.predict_with_details([[0.5]])

        self.assertEqual(predictions.tolist(), ["resonating"])
        self.assertEqual(details[0]["category"], 1)
        self.assertAlmostEqual(details[0]["match"], 0.8)
        self.assertFalse(details[0]["rejected"])

    def test_prediction_rejects_when_no_category_meets_vigilance(self) -> None:
        model = FuzzyARTMAPClassifier(vigilance=1.0).fit(
            [[0.0], [1.0]], ["low", "high"]
        )

        predictions, details = model.predict_with_details([[0.5]])

        self.assertEqual(predictions.tolist(), [None])
        self.assertTrue(details[0]["rejected"])
        self.assertFalse(details[0]["meets_vigilance"])
        self.assertIn(details[0]["label"], {"low", "high"})

    def test_model_state_is_json_friendly_and_round_trips(self) -> None:
        original = FuzzyARTMAPClassifier(vigilance=0.82).fit(self.X, self.y)
        serialized = json.loads(json.dumps(original.to_dict()))

        restored = FuzzyARTMAPClassifier.from_dict(serialized)

        np.testing.assert_array_equal(restored.predict(self.X), original.predict(self.X))
        np.testing.assert_allclose(restored.weights_, original.weights_)
        np.testing.assert_array_equal(
            restored.category_counts_, original.category_counts_
        )

    def test_training_is_deterministic(self) -> None:
        first = FuzzyARTMAPClassifier(vigilance=0.8).fit(self.X, self.y)
        second = FuzzyARTMAPClassifier(vigilance=0.8).fit(self.X, self.y)

        np.testing.assert_allclose(first.weights_, second.weights_)
        self.assertEqual(first.category_labels_, second.category_labels_)

    def test_rejects_non_finite_features_and_wrong_feature_count(self) -> None:
        model = FuzzyARTMAPClassifier().fit(self.X, self.y)

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            model.predict([[np.nan, 0.0]])
        with self.assertRaisesRegex(ValueError, "classifier expects"):
            model.predict([[0.0, 1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
