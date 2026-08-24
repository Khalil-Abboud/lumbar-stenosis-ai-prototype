import unittest
from dataclasses import dataclass

from lumbar_stenosis_ai.splitting import patient_level_split


@dataclass
class Record:
    patient_id: str
    label: str
    slice_id: str


class PatientLevelSplitTests(unittest.TestCase):
    def test_patient_slices_do_not_cross_partitions(self):
        records = [
            Record("p1", "normal", "1"),
            Record("p1", "normal", "2"),
            Record("p2", "normal", "1"),
            Record("p3", "stenosis", "1"),
            Record("p4", "stenosis", "1"),
        ]
        train, test = patient_level_split(records, test_fraction=0.5, seed=7)
        train_patients = {record.patient_id for record in train}
        test_patients = {record.patient_id for record in test}
        self.assertFalse(train_patients & test_patients)
        self.assertEqual(len(train) + len(test), len(records))

    def test_single_patient_cannot_be_split(self):
        with self.assertRaises(ValueError):
            patient_level_split(
                [Record("p1", "normal", "1")],
                test_fraction=0.2,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
