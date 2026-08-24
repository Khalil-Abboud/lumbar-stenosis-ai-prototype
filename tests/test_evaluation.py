import unittest

from lumbar_stenosis_ai.evaluation import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_binary_metrics(self):
        metrics = classification_metrics(["mild", "mild", "severe", "severe"], ["mild", "severe", "severe", "severe"])
        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"]["mild"]["severe"], 1)

    def test_rejects_different_lengths(self):
        with self.assertRaises(ValueError):
            classification_metrics(["a"], [])

    def test_balanced_accuracy_ignores_class_absent_from_y_true(self):
        metrics = classification_metrics(["A", "A"], ["A", "B"])

        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertIn("B", metrics["labels"])
        self.assertEqual(metrics["confusion_matrix"]["A"]["B"], 1)


if __name__ == "__main__":
    unittest.main()
