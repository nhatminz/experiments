#!/usr/bin/env python3
"""Rebuild summary/caption/Figure 1 without running inference."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .plotting import analyze_output
except ImportError:  # direct ``python analyze.py`` compatibility
    from plotting import analyze_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = analyze_output(args.output_dir.resolve())
    print(f"Rendered Figure 1 from {summary['num_interventions']} interventions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
