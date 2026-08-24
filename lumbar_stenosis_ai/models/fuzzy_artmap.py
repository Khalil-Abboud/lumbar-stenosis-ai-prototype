"""A small, dependency-light implementation of supervised Fuzzy ARTMAP.

The implementation follows the core Fuzzy ART operations: min-max feature
scaling, complement coding, fuzzy-AND category choice, vigilance testing and
match tracking after a map-field label mismatch.  A committed category maps
to exactly one class label, which is the usual fast map-learning form used for
classification.

The feature scaler is fitted by :meth:`fit`, or by the first call to
:meth:`partial_fit`, and is then kept fixed.  Keeping it fixed is important:
changing the coordinate system after ART category weights have been learned
would invalidate those weights.  Values outside the calibration range are
clipped by default.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ArrayLike = Sequence[Sequence[float]] | Sequence[float] | np.ndarray


def _as_2d_finite_float_array(X: ArrayLike, *, name: str = "X") -> np.ndarray:
    """Convert an input matrix to a non-empty, finite two-dimensional array."""

    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional feature matrix")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    return values


def _python_scalar(value: Any) -> Any:
    """Return a NumPy scalar as its equivalent Python scalar."""

    return value.item() if isinstance(value, np.generic) else value


class MinMaxFeatureScaler:
    """Scale every feature to ``[0, 1]`` using fixed calibration bounds.

    Constant features are mapped to zero.  With ``clip=True`` (the default),
    later values outside the calibration interval are clipped to the valid
    fuzzy-input range.
    """

    def __init__(self, *, clip: bool = True) -> None:
        self.clip = bool(clip)
        self.feature_min_: np.ndarray | None = None
        self.feature_max_: np.ndarray | None = None
        self.feature_range_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether calibration bounds have been learned."""

        return self.feature_min_ is not None

    def fit(self, X: ArrayLike) -> "MinMaxFeatureScaler":
        """Learn per-feature minimum and maximum values."""

        values = _as_2d_finite_float_array(X)
        self.feature_min_ = values.min(axis=0)
        self.feature_max_ = values.max(axis=0)
        self.feature_range_ = self.feature_max_ - self.feature_min_
        self.n_features_in_ = int(values.shape[1])
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Scale samples using the learned, fixed feature bounds."""

        if not self.is_fitted:
            raise RuntimeError("MinMaxFeatureScaler has not been fitted")

        values = _as_2d_finite_float_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {values.shape[1]} features, but the scaler expects "
                f"{self.n_features_in_}"
            )

        assert self.feature_min_ is not None
        assert self.feature_max_ is not None
        assert self.feature_range_ is not None
        outside_bounds = (values < self.feature_min_) | (values > self.feature_max_)
        if not self.clip and np.any(outside_bounds):
            raise ValueError(
                "X contains values outside the fitted feature range; "
                "enable clipping or use values within the calibration bounds"
            )
        safe_range = np.where(self.feature_range_ == 0.0, 1.0, self.feature_range_)
        scaled = (values - self.feature_min_) / safe_range
        scaled[:, self.feature_range_ == 0.0] = 0.0
        if self.clip:
            scaled = np.clip(scaled, 0.0, 1.0)
        return scaled

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Learn calibration bounds and return scaled samples."""

        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the scaler state."""

        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted scaler")
        assert self.feature_min_ is not None
        assert self.feature_max_ is not None
        return {
            "clip": self.clip,
            "feature_min": self.feature_min_.tolist(),
            "feature_max": self.feature_max_.tolist(),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "MinMaxFeatureScaler":
        """Restore a scaler created by :meth:`to_dict`."""

        if not isinstance(state, Mapping):
            raise ValueError("Scaler state must be a mapping")
        try:
            feature_min = np.asarray(state["feature_min"], dtype=np.float64)
            feature_max = np.asarray(state["feature_max"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid scaler state") from exc
        if (
            feature_min.ndim != 1
            or feature_min.size == 0
            or feature_max.shape != feature_min.shape
            or not np.all(np.isfinite(feature_min))
            or not np.all(np.isfinite(feature_max))
            or np.any(feature_max < feature_min)
        ):
            raise ValueError("Invalid scaler feature bounds")

        scaler = cls(clip=bool(state.get("clip", True)))
        scaler.feature_min_ = feature_min.copy()
        scaler.feature_max_ = feature_max.copy()
        scaler.feature_range_ = feature_max - feature_min
        scaler.n_features_in_ = int(feature_min.size)
        return scaler


class FuzzyARTMAPClassifier:
    """Incremental multiclass classifier based on Fuzzy ARTMAP.

    Parameters
    ----------
    vigilance:
        Baseline similarity required for a category to resonate, in ``[0, 1]``.
        Higher values create more, narrower categories.
    alpha:
        Positive choice-function regularizer.  A small positive value avoids
        division by zero and favors more specific categories.
    beta:
        Category learning rate in ``(0, 1]``.  ``1`` enables standard fast
        learning.
    match_tracking_epsilon:
        Positive increment applied to vigilance after a category predicts the
        wrong supervised label.
    clip:
        Whether the min-max scaler clips later features outside its fitted
        range.  The scaler remains fixed after the first training call.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        vigilance: float = 0.75,
        alpha: float = 1e-3,
        beta: float = 1.0,
        match_tracking_epsilon: float = 1e-6,
        clip: bool = True,
    ) -> None:
        if not 0.0 <= vigilance <= 1.0:
            raise ValueError("vigilance must be in [0, 1]")
        if alpha <= 0.0 or not np.isfinite(alpha):
            raise ValueError("alpha must be a finite value greater than zero")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        if match_tracking_epsilon <= 0.0 or not np.isfinite(match_tracking_epsilon):
            raise ValueError("match_tracking_epsilon must be finite and positive")

        self.vigilance = float(vigilance)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.match_tracking_epsilon = float(match_tracking_epsilon)
        self.clip = bool(clip)

        self.scaler_ = MinMaxFeatureScaler(clip=self.clip)
        self.weights_: np.ndarray | None = None
        self.category_labels_: list[Any] = []
        self.category_counts_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether at least one supervised category has been learned."""

        return self.weights_ is not None and len(self.category_labels_) > 0

    @property
    def n_categories_(self) -> int:
        """Number of committed Fuzzy ART categories."""

        return len(self.category_labels_)

    @property
    def classes_(self) -> np.ndarray:
        """Observed class labels, in deterministic first-seen order."""

        seen: list[Any] = []
        for label in self.category_labels_:
            if label not in seen:
                seen.append(label)
        return np.asarray(seen, dtype=object)

    @staticmethod
    def complement_code(X_scaled: ArrayLike) -> np.ndarray:
        """Return complement-coded fuzzy inputs ``[x, 1 - x]``.

        Inputs must already be in ``[0, 1]``.
        """

        values = _as_2d_finite_float_array(X_scaled, name="X_scaled")
        tolerance = 1e-12
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError("Complement coding requires values in [0, 1]")
        values = np.clip(values, 0.0, 1.0)
        return np.concatenate((values, 1.0 - values), axis=1)

    def fit(self, X: ArrayLike, y: Iterable[Any]) -> "FuzzyARTMAPClassifier":
        """Reset the model and learn categories from a supervised batch."""

        values = _as_2d_finite_float_array(X)
        labels = self._validate_y(y, values.shape[0])

        self.scaler_ = MinMaxFeatureScaler(clip=self.clip)
        scaled = self.scaler_.fit_transform(values)
        self.n_features_in_ = int(values.shape[1])
        self.weights_ = None
        self.category_labels_ = []
        self.category_counts_ = None
        self._learn_encoded(self.complement_code(scaled), labels)
        return self

    def partial_fit(
        self, X: ArrayLike, y: Iterable[Any]
    ) -> "FuzzyARTMAPClassifier":
        """Incrementally learn one or more labeled samples.

        On the first call, this method also calibrates the min-max scaler.  On
        later calls its bounds remain fixed so existing ART weights keep their
        meaning.
        """

        values = _as_2d_finite_float_array(X)
        labels = self._validate_y(y, values.shape[0])

        if not self.scaler_.is_fitted:
            scaled = self.scaler_.fit_transform(values)
            self.n_features_in_ = int(values.shape[1])
        else:
            self._check_feature_count(values)
            scaled = self.scaler_.transform(values)

        self._learn_encoded(self.complement_code(scaled), labels)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict labels, returning ``None`` when no category resonates."""

        predictions, _ = self.predict_with_details(X)
        return predictions

    def predict_with_details(
        self, X: ArrayLike
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Predict labels and expose the winning category's ART diagnostics.

        The detail dictionaries contain ``category``, ``choice``, ``match`` and
        ``meets_vigilance``.  Categories are searched by stable descending
        choice value, and the first one meeting vigilance is accepted.  If no
        category resonates, the prediction is ``None`` and diagnostics describe
        the best-choice candidate with ``rejected=True``.
        """

        self._require_fitted()
        values = _as_2d_finite_float_array(X)
        self._check_feature_count(values)
        encoded = self.complement_code(self.scaler_.transform(values))

        predictions: list[Any] = []
        details: list[dict[str, Any]] = []
        for sample in encoded:
            choices = self._choice_values(sample)
            search_order = np.argsort(-choices, kind="stable")
            category: int | None = None
            for candidate in search_order:
                candidate_index = int(candidate)
                candidate_match = self._match_value(sample, candidate_index)
                if candidate_match + 1e-12 >= self.vigilance:
                    category = candidate_index
                    break

            if category is None:
                rejected = True
                # Keep the strongest candidate's diagnostics so callers can
                # inspect how close the rejected input was to resonance.
                category = int(search_order[0])
            else:
                rejected = False

            match = self._match_value(sample, category)
            label = self.category_labels_[category]
            predictions.append(None if rejected else label)
            details.append(
                {
                    "category": category,
                    "label": label,
                    "choice": float(choices[category]),
                    "match": float(match),
                    "meets_vigilance": not rejected,
                    "rejected": rejected,
                }
            )
        return np.asarray(predictions, dtype=object), details

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly model state for reproducible persistence."""

        self._require_fitted()
        assert self.weights_ is not None
        assert self.category_counts_ is not None

        labels = [_python_scalar(label) for label in self.category_labels_]
        allowed_label_types = (str, int, float, bool)
        if any(label is None or not isinstance(label, allowed_label_types) for label in labels):
            raise TypeError(
                "Only string, integer, float, or boolean labels can be serialized"
            )

        return {
            "model": self.__class__.__name__,
            "version": self.STATE_VERSION,
            "parameters": {
                "vigilance": self.vigilance,
                "alpha": self.alpha,
                "beta": self.beta,
                "match_tracking_epsilon": self.match_tracking_epsilon,
                "clip": self.clip,
            },
            "scaler": self.scaler_.to_dict(),
            "state": {
                "n_features_in": self.n_features_in_,
                "weights": self.weights_.tolist(),
                "category_labels": labels,
                "category_counts": self.category_counts_.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FuzzyARTMAPClassifier":
        """Restore a fitted classifier created by :meth:`to_dict`."""

        if not isinstance(payload, Mapping):
            raise ValueError("Model payload must be a mapping")
        if payload.get("model") != cls.__name__:
            raise ValueError("Payload is not a FuzzyARTMAPClassifier state")
        if payload.get("version") != cls.STATE_VERSION:
            raise ValueError("Unsupported FuzzyARTMAPClassifier state version")

        try:
            parameters = payload["parameters"]
            model_state = payload["state"]
            model = cls(**dict(parameters))
            scaler = MinMaxFeatureScaler.from_dict(payload["scaler"])
            weights = np.asarray(model_state["weights"], dtype=np.float64)
            category_labels = [
                _python_scalar(label) for label in model_state["category_labels"]
            ]
            category_counts = np.asarray(
                model_state["category_counts"], dtype=np.int64
            )
            n_features = int(model_state["n_features_in"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid FuzzyARTMAPClassifier state") from exc

        expected_width = 2 * n_features
        if (
            n_features <= 0
            or weights.ndim != 2
            or weights.shape[0] == 0
            or weights.shape[1] != expected_width
            or len(category_labels) != weights.shape[0]
            or category_counts.shape != (weights.shape[0],)
            or np.any(category_counts <= 0)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.any(weights > 1.0)
            or scaler.n_features_in_ != n_features
        ):
            raise ValueError("Inconsistent FuzzyARTMAPClassifier state")

        model.scaler_ = scaler
        model.n_features_in_ = n_features
        model.weights_ = weights.copy()
        model.category_labels_ = category_labels
        model.category_counts_ = category_counts.copy()
        return model

    def _validate_y(self, y: Iterable[Any], expected_length: int) -> list[Any]:
        if isinstance(y, (str, bytes)):
            raise ValueError("y must contain one label per sample")
        try:
            raw_labels = list(y)
        except TypeError as exc:
            raise ValueError("y must be a one-dimensional iterable of labels") from exc
        if len(raw_labels) != expected_length:
            raise ValueError(
                f"X contains {expected_length} samples, but y contains "
                f"{len(raw_labels)} labels"
            )

        labels: list[Any] = []
        for raw_label in raw_labels:
            label = _python_scalar(raw_label)
            if label is None or isinstance(label, (list, dict, set, np.ndarray)):
                raise ValueError("Each target label must be a non-null scalar")
            try:
                hash(label)
            except TypeError as exc:
                raise ValueError("Each target label must be hashable") from exc
            if isinstance(label, float) and not np.isfinite(label):
                raise ValueError("Target labels must not be NaN or infinite")
            labels.append(label)
        return labels

    def _learn_encoded(self, encoded: np.ndarray, labels: list[Any]) -> None:
        for sample, label in zip(encoded, labels):
            self._learn_one(sample, label)

    def _learn_one(self, sample: np.ndarray, label: Any) -> None:
        if self.weights_ is None:
            self._commit_category(sample, label)
            return

        choices = self._choice_values(sample)
        # Stable sorting makes category selection deterministic when tied.
        search_order = np.argsort(-choices, kind="stable")
        current_vigilance = self.vigilance

        for category in search_order:
            category_index = int(category)
            match = self._match_value(sample, category_index)
            if match + 1e-12 < current_vigilance:
                continue

            if self.category_labels_[category_index] == label:
                fuzzy_intersection = np.minimum(
                    sample, self.weights_[category_index]
                )
                self.weights_[category_index] = (
                    self.beta * fuzzy_intersection
                    + (1.0 - self.beta) * self.weights_[category_index]
                )
                assert self.category_counts_ is not None
                self.category_counts_[category_index] += 1
                return

            # Map-field mismatch: raise vigilance just above this category's
            # match, reset it, and continue the category search.
            current_vigilance = match + self.match_tracking_epsilon

        self._commit_category(sample, label)

    def _commit_category(self, sample: np.ndarray, label: Any) -> None:
        if self.weights_ is None:
            self.weights_ = sample.reshape(1, -1).copy()
            self.category_counts_ = np.ones(1, dtype=np.int64)
        else:
            self.weights_ = np.vstack((self.weights_, sample))
            assert self.category_counts_ is not None
            self.category_counts_ = np.append(self.category_counts_, 1)
        self.category_labels_.append(label)

    def _choice_values(self, sample: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None
        fuzzy_intersections = np.minimum(self.weights_, sample)
        numerators = fuzzy_intersections.sum(axis=1)
        denominators = self.alpha + self.weights_.sum(axis=1)
        return numerators / denominators

    def _match_value(self, sample: np.ndarray, category: int) -> float:
        assert self.weights_ is not None
        input_norm = float(sample.sum())
        if input_norm <= 0.0:  # Complement-coded inputs should never reach this.
            return 0.0
        intersection_norm = float(
            np.minimum(sample, self.weights_[category]).sum()
        )
        return intersection_norm / input_norm

    def _check_feature_count(self, values: np.ndarray) -> None:
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {values.shape[1]} features, but the classifier expects "
                f"{self.n_features_in_}"
            )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("FuzzyARTMAPClassifier has not been fitted")
