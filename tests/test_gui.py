import tempfile
import unittest
from pathlib import Path

from PIL import Image

from lumbar_stenosis_ai.gui import (
    default_output_directory,
    format_prediction_ar,
    format_prediction_ru,
    validation_summary,
)


class GuiHelperTests(unittest.TestCase):
    def test_default_output_is_a_new_run_below_project(self):
        project = Path("example_project").resolve()
        output = default_output_directory(project)
        self.assertEqual(output.parent, project / "runs")
        self.assertTrue(output.name.startswith("multidisorder_t2_gui_"))

    def test_validation_summary_uses_existing_manifest_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.png"
            Image.new("L", (8, 8), color=42).save(image)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,patient_id,image_path,label,split,plane,level,sequence\n"
                "s1,p1,sample.png,stenosis_present,,sagittal,whole-lumbar,T2\n",
                encoding="utf-8",
            )

            summary = validation_summary(manifest)

            self.assertEqual(summary["status"], "valid")
            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(summary["patient_count"], 1)
            self.assertEqual(summary["sequence"], ["T2"])

    def test_prediction_summary_is_available_in_both_languages(self):
        result = {
            "predicted_label": "stenosis_present",
            "candidate_label": "stenosis_present",
            "match": 0.8,
            "requires_review": False,
            "image_path": "sample.png",
        }
        self.assertIn("имеется разметка стеноза", format_prediction_ru(result))
        self.assertIn("توجد علامة تضيق", format_prediction_ar(result))


if __name__ == "__main__":
    unittest.main()
