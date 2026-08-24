"""ResNet18 feature extraction for raster images and individual DICOM slices."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lumbar_stenosis_ai.data.image_io import load_image


def _require_torch():
    try:
        import torch
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Feature extraction requires torch and torchvision. "
            "Install the packages listed in requirements.txt."
        ) from exc
    return torch, nn, ResNet18_Weights, resnet18


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    paths: tuple[Path, ...]
    features: np.ndarray


class ResNet18FeatureExtractor:
    """Convert 2-D images to reproducible 512-dimensional feature vectors."""

    feature_count = 512

    def __init__(
        self, *, pretrained: bool = True, device: str = "auto", seed: int = 42
    ) -> None:
        torch, nn, weights_type, resnet18 = _require_torch()
        self._torch = torch
        self.pretrained = bool(pretrained)
        self.seed = int(seed)
        self.device = self._resolve_device(device)

        weights = weights_type.DEFAULT if self.pretrained else None
        # torchvision initializes the module before loading pretrained weights.
        # Isolating and seeding that initialization also makes the explicitly
        # requested non-pretrained feature space reproducible after reload.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self._model = resnet18(weights=weights)
        self._model.fc = nn.Identity()
        self._model.eval()
        self._model.to(self.device)

        self.weights_id = (
            f"{weights_type.__name__}.{weights.name}"
            if weights is not None
            else f"random_initialization_seed_{self.seed}"
        )
        self.preprocessing_id = (
            f"{self.weights_id}.transforms"
            if weights is not None
            else "resize_224_imagenet_mean_std"
        )
        self.state_sha256 = self._state_sha256()

        if weights is not None:
            self._preprocess = weights.transforms()
        else:
            from torchvision import transforms

            self._preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

    def _state_sha256(self) -> str:
        """Fingerprint the exact feature coordinate system persisted by a run."""

        digest = hashlib.sha256()
        for name, tensor in self._model.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            values = tensor.detach().cpu().contiguous().numpy()
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(str(values.shape).encode("ascii"))
            digest.update(values.tobytes())
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Describe the exact backbone and preprocessing used for a bundle."""

        return {
            "name": "ResNet18",
            "output_features": self.feature_count,
            "pretrained_imagenet": self.pretrained,
            "initialization_seed": self.seed,
            "resolved_device": str(self.device),
            "weights_id": self.weights_id,
            "preprocessing_id": self.preprocessing_id,
            "state_sha256": self.state_sha256,
        }

    def _resolve_device(self, requested: str):
        if requested == "auto":
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not self._torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
        return self._torch.device(requested)

    def extract(self, paths: Sequence[str | Path], *, batch_size: int = 8) -> ExtractionResult:
        """Extract features while preserving the input order."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        normalized_paths = tuple(Path(path).resolve() for path in paths)
        if not normalized_paths:
            return ExtractionResult(normalized_paths, np.empty((0, self.feature_count), dtype=np.float32))

        feature_batches: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(normalized_paths), batch_size):
                batch_paths = normalized_paths[start : start + batch_size]
                tensors = [self._preprocess(load_image(path)) for path in batch_paths]
                inputs = self._torch.stack(tensors).to(self.device)
                outputs = self._model(inputs)
                feature_batches.append(outputs.detach().cpu().numpy().astype(np.float32, copy=False))

        features = np.concatenate(feature_batches, axis=0)
        if features.shape != (len(normalized_paths), self.feature_count):
            raise RuntimeError(f"Unexpected ResNet18 feature shape: {features.shape}.")
        return ExtractionResult(normalized_paths, features)

    def iter_extract(
        self, paths: Iterable[str | Path], *, batch_size: int = 8
    ) -> Iterable[tuple[Path, np.ndarray]]:
        result = self.extract(tuple(paths), batch_size=batch_size)
        return zip(result.paths, result.features, strict=True)
