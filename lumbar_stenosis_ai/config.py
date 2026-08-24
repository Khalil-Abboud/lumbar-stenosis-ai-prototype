"""Configuration objects shared by training and inference pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Serializable configuration for the article-2 software prototype."""

    seed: int = 42
    test_fraction: float = 0.2
    device: str = "auto"
    batch_size: int = 8
    pretrained: bool = True
    art_choice: float = 0.001
    art_learning_rate: float = 1.0
    art_vigilance: float = 0.75
    art_match_tracking: float = 0.001

    def __post_init__(self) -> None:
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1).")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.art_choice <= 0:
            raise ValueError("art_choice must be positive.")
        if not 0.0 < self.art_learning_rate <= 1.0:
            raise ValueError("art_learning_rate must be in (0, 1].")
        if not 0.0 <= self.art_vigilance <= 1.0:
            raise ValueError("art_vigilance must be in [0, 1].")
        if self.art_match_tracking <= 0:
            raise ValueError("art_match_tracking must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PipelineConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})
