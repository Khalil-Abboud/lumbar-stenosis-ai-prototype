"""Image readers for ordinary raster files and individual DICOM slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError


RASTER_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
DICOM_EXTENSIONS = frozenset({".dcm", ".dicom", ".ima"})


class ImageLoadingError(ValueError):
    """Base class for an image that exists but cannot be decoded safely."""


class UnsupportedImageFormatError(ImageLoadingError):
    """Raised when a path is neither a supported raster nor DICOM file."""


class DicomDependencyError(ImportError):
    """Raised when DICOM input is requested without the optional dependency."""


class DicomImageError(ImageLoadingError):
    """Raised for malformed, compressed-without-codec, or unsupported DICOM."""


def _existing_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"Image file does not exist: {resolved}")
    return resolved


def load_raster_image(path: str | Path) -> Image.Image:
    """Read a common raster image and return a detached RGB PIL image."""

    image_path = _existing_file(path)
    try:
        with Image.open(image_path) as source:
            source.load()
            return source.convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadingError(
            f"Could not decode raster image {image_path}: {exc}"
        ) from exc


def _import_pydicom() -> Any:
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        if exc.name != "pydicom":
            raise
        raise DicomDependencyError(
            "Reading DICOM images requires the optional 'pydicom' package. "
            "Install the project's DICOM dependencies (or run: pip install pydicom)."
        ) from exc
    return pydicom


def _lut_functions() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    """Return modality and VOI LUT helpers across supported pydicom versions."""

    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut

        return apply_modality_lut, apply_voi_lut
    except (ImportError, AttributeError):
        try:
            from pydicom.pixel_data_handlers.util import (
                apply_modality_lut,
                apply_voi_lut,
            )

            return apply_modality_lut, apply_voi_lut
        except (ImportError, AttributeError):
            return None, None


def _first_number(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    else:
        try:
            # pydicom's MultiValue supports indexing but is not a list.
            if not isinstance(value, (str, bytes)) and len(value) > 0:
                value = value[0]
        except (TypeError, AttributeError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_modality_transform(
    pixels: np.ndarray, dataset: Any, helper: Callable[..., Any] | None
) -> np.ndarray:
    if helper is not None:
        try:
            return np.asarray(helper(pixels, dataset), dtype=np.float32)
        except (AttributeError, IndexError, TypeError, ValueError):
            # Some real-world files contain incomplete LUT metadata.  Rescale
            # tags remain a safe fallback and cover typical MR/CT slices.
            pass

    slope = _first_number(getattr(dataset, "RescaleSlope", None), 1.0)
    intercept = _first_number(getattr(dataset, "RescaleIntercept", None), 0.0)
    return np.asarray(pixels, dtype=np.float32) * slope + intercept


def _manual_window(pixels: np.ndarray, dataset: Any) -> np.ndarray:
    center = _first_number(getattr(dataset, "WindowCenter", None), float("nan"))
    width = _first_number(getattr(dataset, "WindowWidth", None), float("nan"))
    if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
        return pixels

    # DICOM PS3.3 C.11.2 linear windowing.  Clipping here is sufficient
    # because the result is normalized to 8-bit below.
    if width <= 1:
        return np.where(pixels <= center - 0.5, 0.0, 1.0)
    lower = center - 0.5 - (width - 1.0) / 2.0
    upper = center - 0.5 + (width - 1.0) / 2.0
    return np.clip(pixels, lower, upper)


def _apply_voi_transform(
    pixels: np.ndarray, dataset: Any, helper: Callable[..., Any] | None
) -> np.ndarray:
    has_voi = bool(getattr(dataset, "VOILUTSequence", None)) or (
        getattr(dataset, "WindowCenter", None) is not None
        and getattr(dataset, "WindowWidth", None) is not None
    )
    if not has_voi:
        return pixels

    if helper is not None:
        try:
            return np.asarray(helper(pixels, dataset, index=0), dtype=np.float32)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    return _manual_window(pixels, dataset)


def _normalize_uint8(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise DicomImageError("DICOM pixel data contains no finite values")

    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)

    scaled = (np.clip(values, low, high) - low) * (255.0 / (high - low))
    return np.rint(scaled).astype(np.uint8)


def _dicom_pixels(dataset: Any, path: Path) -> np.ndarray:
    if "PixelData" not in dataset:
        raise DicomImageError(f"DICOM file has no PixelData element: {path}")
    try:
        return np.asarray(dataset.pixel_array)
    except Exception as exc:  # pydicom uses codec-specific exception classes
        raise DicomImageError(
            f"Could not decode DICOM pixel data in {path}. "
            "The file may be malformed or require an optional compression codec: "
            f"{exc}"
        ) from exc


def _positive_dicom_integer(dataset: Any, attribute: str, default: int) -> int:
    raw_value = getattr(dataset, attribute, None)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise DicomImageError(
            f"DICOM attribute {attribute} must be a positive integer, "
            f"got {raw_value!r}"
        ) from exc
    if value < 1:
        raise DicomImageError(
            f"DICOM attribute {attribute} must be a positive integer, got {value}"
        )
    return value


def load_dicom_image(path: str | Path) -> Image.Image:
    """Read one DICOM slice and convert it to an 8-bit RGB PIL image.

    The conversion applies the modality rescale/LUT, then the VOI LUT or
    window when present.  MONOCHROME1 is inverted after normalization so its
    display polarity matches ordinary images.  Multi-frame volumes are
    rejected explicitly; callers should select a slice before using this
    single-slice API.
    """

    image_path = _existing_file(path)
    pydicom = _import_pydicom()
    try:
        dataset = pydicom.dcmread(str(image_path))
    except Exception as exc:
        raise DicomImageError(f"Could not read DICOM file {image_path}: {exc}") from exc

    modality = str(getattr(dataset, "Modality", "")).strip().upper()
    if not modality:
        raise DicomImageError(
            f"DICOM Modality is missing in {image_path}; only MR images are supported"
        )
    if modality != "MR":
        raise DicomImageError(
            f"Unsupported DICOM Modality {modality!r} in {image_path}; "
            "only MR images are supported"
        )

    number_of_frames = _positive_dicom_integer(dataset, "NumberOfFrames", 1)
    if number_of_frames != 1:
        raise DicomImageError(
            f"Expected a single-frame DICOM slice, but NumberOfFrames is "
            f"{number_of_frames} in {image_path}. Select one frame explicitly "
            "before using this reader."
        )

    samples_per_pixel = _positive_dicom_integer(dataset, "SamplesPerPixel", 1)
    pixels = _dicom_pixels(dataset, image_path)

    # Colour is determined from DICOM metadata, never guessed from an array
    # dimension that could equally represent multiple grayscale frames.
    if samples_per_pixel == 3:
        if pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise DicomImageError(
                "SamplesPerPixel declares a colour image, but decoded pixel "
                f"shape is {pixels.shape} in {image_path}"
            )
        # pydicom normally converts YBR pixel data to RGB while decoding.
        color = _normalize_uint8(pixels)
        return Image.fromarray(color).convert("RGB")

    if samples_per_pixel != 1:
        raise DicomImageError(
            f"Unsupported SamplesPerPixel value {samples_per_pixel} in {image_path}; "
            "only grayscale (1) and RGB/YBR colour (3) slices are supported"
        )

    # A few encoders retain a leading singleton frame dimension even when
    # NumberOfFrames is one.  It is unambiguous and safe to remove only here.
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]

    if pixels.ndim != 2:
        raise DicomImageError(
            "SamplesPerPixel declares a grayscale slice, but decoded pixel "
            f"shape is {pixels.shape} in {image_path}; multi-frame input must "
            "be sliced explicitly"
        )

    modality_lut, voi_lut = _lut_functions()
    display_pixels = _apply_modality_transform(pixels, dataset, modality_lut)
    display_pixels = _apply_voi_transform(display_pixels, dataset, voi_lut)
    grayscale = _normalize_uint8(display_pixels)

    photometric = str(
        getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")
    ).upper()
    if photometric == "MONOCHROME1":
        grayscale = np.uint8(255) - grayscale

    return Image.fromarray(grayscale).convert("RGB")


def load_image(path: str | Path) -> Image.Image:
    """Load a raster image or DICOM slice based on its file extension."""

    image_path = Path(path)
    suffix = image_path.suffix.casefold()
    if suffix in DICOM_EXTENSIONS:
        return load_dicom_image(image_path)
    if suffix in RASTER_EXTENSIONS:
        return load_raster_image(image_path)
    raise UnsupportedImageFormatError(
        f"Unsupported image extension {suffix or '<none>'!r} for {image_path}. "
        f"Raster: {', '.join(sorted(RASTER_EXTENSIONS))}; "
        f"DICOM: {', '.join(sorted(DICOM_EXTENSIONS))}."
    )


read_image = load_image


__all__ = [
    "DICOM_EXTENSIONS",
    "RASTER_EXTENSIONS",
    "DicomDependencyError",
    "DicomImageError",
    "ImageLoadingError",
    "UnsupportedImageFormatError",
    "load_dicom_image",
    "load_image",
    "load_raster_image",
    "read_image",
]
