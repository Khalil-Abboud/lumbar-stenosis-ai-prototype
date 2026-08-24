from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from lumbar_stenosis_ai.data.image_io import (
    DicomDependencyError,
    DicomImageError,
    UnsupportedImageFormatError,
    load_dicom_image,
    load_image,
    load_raster_image,
)


class RasterImageTests(unittest.TestCase):
    def test_raster_reader_returns_detached_rgb_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grayscale.png"
            Image.new("L", (3, 2), color=127).save(path)

            image = load_raster_image(path)
            routed_image = load_image(path)

            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (3, 2))
            self.assertEqual(routed_image.mode, "RGB")
            path.unlink()  # proves the returned PIL object no longer owns the file
            self.assertEqual(image.getpixel((0, 0)), (127, 127, 127))

    def test_unknown_extension_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedImageFormatError):
            load_image("image.unknown")


class _FakeDicomDataset(dict):
    def __init__(
        self,
        pixels: np.ndarray,
        *,
        number_of_frames: int | str | None = None,
        samples_per_pixel: int | str = 1,
        photometric: str = "MONOCHROME2",
        modality: str | None = "MR",
    ) -> None:
        super().__init__(PixelData=b"present")
        self.pixel_array = pixels
        self.SamplesPerPixel = samples_per_pixel
        self.PhotometricInterpretation = photometric
        if modality is not None:
            self.Modality = modality
        if number_of_frames is not None:
            self.NumberOfFrames = number_of_frames


class DicomMetadataValidationTests(unittest.TestCase):
    @staticmethod
    def _reader_for(dataset: _FakeDicomDataset) -> SimpleNamespace:
        return SimpleNamespace(dcmread=lambda _path: dataset)

    def test_explicit_multiframe_grayscale_is_rejected(self) -> None:
        dataset = _FakeDicomDataset(
            np.zeros((2, 2, 2), dtype=np.uint16),
            number_of_frames="2",
            samples_per_pixel=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiframe.dcm"
            path.touch()
            with patch(
                "lumbar_stenosis_ai.data.image_io._import_pydicom",
                return_value=self._reader_for(dataset),
            ):
                with self.assertRaisesRegex(DicomImageError, "single-frame"):
                    load_dicom_image(path)

    def test_three_dimensional_grayscale_is_not_inferred_as_colour(self) -> None:
        dataset = _FakeDicomDataset(
            np.zeros((2, 2, 3), dtype=np.uint16), samples_per_pixel=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.dcm"
            path.touch()
            with patch(
                "lumbar_stenosis_ai.data.image_io._import_pydicom",
                return_value=self._reader_for(dataset),
            ):
                with self.assertRaisesRegex(DicomImageError, "declares a grayscale"):
                    load_dicom_image(path)

    def test_single_frame_colour_requires_samples_per_pixel_metadata(self) -> None:
        pixels = np.asarray(
            [[[0, 64, 128], [255, 128, 0]]], dtype=np.uint8
        )
        dataset = _FakeDicomDataset(
            pixels,
            number_of_frames=1,
            samples_per_pixel=3,
            photometric="RGB",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "colour.dcm"
            path.touch()
            with patch(
                "lumbar_stenosis_ai.data.image_io._import_pydicom",
                return_value=self._reader_for(dataset),
            ):
                image = load_dicom_image(path)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (2, 1))

    def test_missing_or_non_mr_modality_is_rejected(self) -> None:
        datasets = (
            (_FakeDicomDataset(np.zeros((2, 2)), modality=None), "missing"),
            (_FakeDicomDataset(np.zeros((2, 2)), modality="CT"), "only MR"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-modality.dcm"
            path.touch()
            for dataset, expected_message in datasets:
                with self.subTest(expected_message=expected_message), patch(
                    "lumbar_stenosis_ai.data.image_io._import_pydicom",
                    return_value=self._reader_for(dataset),
                ):
                    with self.assertRaisesRegex(DicomImageError, expected_message):
                        load_dicom_image(path)


@unittest.skipUnless(
    importlib.util.find_spec("pydicom") is not None,
    "pydicom is an optional dependency",
)
class DicomImageTests(unittest.TestCase):
    def test_monochrome1_is_rescaled_windowed_and_inverted(self) -> None:
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slice.dcm"
            file_meta = FileMetaDataset()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
            dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
            dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            dataset.Rows = 2
            dataset.Columns = 2
            dataset.SamplesPerPixel = 1
            dataset.Modality = "MR"
            dataset.PhotometricInterpretation = "MONOCHROME1"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.RescaleSlope = 2
            dataset.RescaleIntercept = -100
            dataset.WindowCenter = 900
            dataset.WindowWidth = 2000
            dataset.PixelData = np.asarray(
                [[0, 250], [500, 1000]], dtype=np.uint16
            ).tobytes()
            pydicom.dcmwrite(str(path), dataset, enforce_file_format=True)

            image = load_dicom_image(path)

            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (2, 2))
            self.assertGreater(image.getpixel((0, 0))[0], image.getpixel((1, 1))[0])


class MissingDicomDependencyTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("pydicom") is None,
        "only applies when the optional dependency is absent",
    )
    def test_missing_dependency_has_installation_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slice.dcm"
            path.touch()
            with self.assertRaisesRegex(DicomDependencyError, "pip install pydicom"):
                load_dicom_image(path)


if __name__ == "__main__":
    unittest.main()
