from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lumbar_stenosis_ai.config import PipelineConfig
from lumbar_stenosis_ai.pipeline import predict_images, train_pipeline, update_pipeline


HAS_INTEGRATION_DEPENDENCIES = all(
    importlib.util.find_spec(package) is not None
    for package in ("pydicom", "torch", "torchvision")
)


@unittest.skipUnless(
    HAS_INTEGRATION_DEPENDENCIES,
    "DICOM-to-ResNet integration dependencies are not installed",
)
class DicomPipelineIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_mr_slice(path: Path, offset: int) -> None:
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = "MR"
        dataset.Rows = 16
        dataset.Columns = 16
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        base = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
        pattern = (np.indices((16, 16)).sum(axis=0) % (offset + 2)).astype(
            np.uint16
        )
        pixels = np.roll(base, shift=offset, axis=0) + pattern * np.uint16(50)
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(str(path), enforce_file_format=True)

    def test_dicom_to_resnet_artmap_bundle_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filenames = (
                "a_train.dcm",
                "b_train.dcm",
                "a_test.dcm",
                "b_test.dcm",
                "c_update.dcm",
            )
            for offset, filename in enumerate(filenames, start=1):
                self._write_mr_slice(root / filename, offset * 100)

            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,patient_id,image_path,label,split,plane,level,sequence\n"
                "a1,p1,a_train.dcm,a,train,axial,L4-L5,T2\n"
                "b1,p2,b_train.dcm,b,train,axial,L4-L5,T2\n"
                "a2,p3,a_test.dcm,a,test,axial,L4-L5,T2\n"
                "b2,p4,b_test.dcm,b,test,axial,L4-L5,T2\n",
                encoding="utf-8",
            )
            bundle = root / "bundle"
            result = train_pipeline(
                manifest,
                bundle,
                config=PipelineConfig(
                    pretrained=False,
                    device="cpu",
                    batch_size=2,
                    seed=17,
                ),
            )

            self.assertEqual(result.train_count, 2)
            self.assertEqual(result.test_count, 2)
            self.assertTrue((bundle / "metadata.json").is_file())
            self.assertTrue((bundle / "manifest_snapshot.csv").is_file())

            prediction = predict_images(bundle, [root / "a_train.dcm"])[0]
            self.assertEqual(prediction["predicted_label"], "a")
            self.assertFalse(prediction["rejected"])
            self.assertTrue(prediction["research_use_only"])

            update_manifest = root / "update.csv"
            update_manifest.write_text(
                "sample_id,patient_id,image_path,label,split,plane,level,sequence\n"
                "c1,p5,c_update.dcm,c,,axial,L4-L5,T2\n",
                encoding="utf-8",
            )
            updated_bundle = root / "updated_bundle"
            update_result = update_pipeline(
                bundle, update_manifest, updated_bundle
            )
            self.assertEqual(update_result.sample_count, 1)
            self.assertGreaterEqual(
                update_result.updated_category_count,
                update_result.previous_category_count,
            )
            updated_prediction = predict_images(
                updated_bundle, [root / "c_update.dcm"]
            )[0]
            self.assertIn("predicted_label", updated_prediction)
            self.assertIn(
                "out_of_training_range_value_count", updated_prediction
            )


if __name__ == "__main__":
    unittest.main()
