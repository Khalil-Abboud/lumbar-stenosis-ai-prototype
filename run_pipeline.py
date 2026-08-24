"""Backward-compatible launcher for the package command-line interface."""

from lumbar_stenosis_ai.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
