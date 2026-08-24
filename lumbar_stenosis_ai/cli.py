"""Command-line interface for the article-2 research software."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lumbar_stenosis_ai.config import PipelineConfig
from lumbar_stenosis_ai.data import load_manifest
from lumbar_stenosis_ai.evaluation import classification_metrics
from lumbar_stenosis_ai.models import FuzzyARTMAPClassifier
from lumbar_stenosis_ai.pipeline import (
    extract_manifest_features,
    predict_images,
    train_pipeline,
    update_pipeline,
)


DISCLAIMER = (
    "RESEARCH PROTOTYPE ONLY - not a medical device and not for diagnosis "
    "or treatment decisions."
)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained ResNet18 weights (default: true).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumbar-stenosis-ai",
        description="ResNet18 + Fuzzy ART-MAP research pipeline for lumbar MRI slices.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate an anonymized CSV manifest.")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--allow-empty-labels", action="store_true")

    extract = subparsers.add_parser("extract", help="Extract and save 512 ResNet18 features.")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--seed", type=int, default=42)
    _add_runtime_arguments(extract)

    train = subparsers.add_parser(
        "train", help="Train ART-MAP and evaluate only on an unseen patient split."
    )
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.add_argument("--vigilance", type=float, default=0.75)
    train.add_argument("--choice", type=float, default=0.001)
    train.add_argument("--learning-rate", type=float, default=1.0)
    train.add_argument("--match-tracking", type=float, default=0.001)
    train.add_argument("--save-features", action=argparse.BooleanOptionalAction, default=True)
    _add_runtime_arguments(train)

    update = subparsers.add_parser(
        "update", help="Incrementally update a saved ART-MAP model with new labeled cases."
    )
    update.add_argument("--model-dir", required=True, type=Path)
    update.add_argument("--manifest", required=True, type=Path)
    update.add_argument("--output-dir", required=True, type=Path)
    update.add_argument("--save-features", action=argparse.BooleanOptionalAction, default=True)

    predict = subparsers.add_parser(
        "predict", help="Predict labels for one or more raster/DICOM slices."
    )
    predict.add_argument("--model-dir", required=True, type=Path)
    predict.add_argument("--image", required=True, action="append", type=Path)

    subparsers.add_parser(
        "self-check", help="Run a fast synthetic test of ART-MAP without medical claims."
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        seed=args.seed,
        test_fraction=args.test_fraction,
        device=args.device,
        batch_size=args.batch_size,
        pretrained=args.pretrained,
        art_choice=args.choice,
        art_learning_rate=args.learning_rate,
        art_vigilance=args.vigilance,
        art_match_tracking=args.match_tracking,
    )


def _validate_command(args: argparse.Namespace) -> dict[str, Any]:
    records = load_manifest(args.manifest, require_labels=not args.allow_empty_labels)
    return {
        "manifest": str(args.manifest.resolve()),
        "sample_count": len(records),
        "patient_count": len({record.patient_id for record in records}),
        "labels": dict(sorted(Counter(record.label for record in records).items())),
        "splits": dict(sorted(Counter(record.split or "automatic" for record in records).items())),
        "planes": dict(sorted(Counter(record.plane or "unspecified" for record in records).items())),
        "levels": dict(sorted(Counter(record.level or "unspecified" for record in records).items())),
        "sequences": dict(
            sorted(Counter(record.sequence or "unspecified" for record in records).items())
        ),
        "status": "valid",
    }


def _self_check() -> dict[str, Any]:
    # Two deliberately separated synthetic clusters.  This validates the ART
    # mechanics only; it is not an MRI experiment or a clinical result.
    train_x = np.asarray(
        [[0.08, 0.12], [0.12, 0.08], [0.10, 0.11], [0.88, 0.92], [0.92, 0.88], [0.90, 0.91]],
        dtype=np.float64,
    )
    train_y = np.asarray(["class_a"] * 3 + ["class_b"] * 3, dtype=object)
    test_x = np.asarray([[0.09, 0.10], [0.91, 0.90]], dtype=np.float64)
    test_y = ["class_a", "class_b"]
    model = FuzzyARTMAPClassifier(vigilance=0.75)
    model.fit(train_x, train_y)
    predictions, details = model.predict_with_details(test_x)
    restored = FuzzyARTMAPClassifier.from_dict(model.to_dict())
    restored_predictions = restored.predict(test_x)
    if predictions.tolist() != restored_predictions.tolist():
        raise RuntimeError("Model serialization self-check failed.")
    return {
        "status": "passed",
        "synthetic_only": True,
        "category_count": model.n_categories_,
        "predictions": predictions.tolist(),
        "details": details,
        "metrics": classification_metrics(test_y, predictions.tolist()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(DISCLAIMER, file=sys.stderr)
    try:
        if args.command == "validate":
            result: Any = _validate_command(args)
        elif args.command == "extract":
            output = extract_manifest_features(
                args.manifest,
                args.output,
                pretrained=args.pretrained,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            result = {"status": "completed", "features_file": str(output.resolve())}
        elif args.command == "train":
            training = train_pipeline(
                args.manifest,
                args.output_dir,
                config=_config_from_args(args),
                save_features=args.save_features,
            )
            result = {
                "status": "completed",
                "output_dir": str(training.output_dir),
                "train_count": training.train_count,
                "test_count": training.test_count,
                "artmap_category_count": training.category_count,
                "metrics": training.metrics,
            }
        elif args.command == "update":
            update_result = update_pipeline(
                args.model_dir,
                args.manifest,
                args.output_dir,
                save_features=args.save_features,
            )
            result = {
                "status": "completed",
                "output_dir": str(update_result.output_dir),
                "incremental_sample_count": update_result.sample_count,
                "previous_artmap_category_count": update_result.previous_category_count,
                "updated_artmap_category_count": update_result.updated_category_count,
                "out_of_training_range_value_count": (
                    update_result.out_of_training_range_value_count
                ),
                "scaler_bounds_remained_fixed": True,
            }
        elif args.command == "predict":
            result = {
                "status": "completed",
                "predictions": predict_images(args.model_dir, args.image),
                "match_is_not_probability": True,
            }
        elif args.command == "self-check":
            result = _self_check()
        else:  # pragma: no cover - argparse enforces the command
            parser.error(f"Unknown command: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
