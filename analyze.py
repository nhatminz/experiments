#!/usr/bin/env python3
"""Rebuild summary/caption/Figure 1 without running inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

try:
    from .plotting import analyze_output
    from .accuracy_analysis import analyze_accuracy
except ImportError:  # direct ``python analyze.py`` compatibility
    from plotting import analyze_output
    from accuracy_analysis import analyze_accuracy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = analyze_output(args.output_dir.resolve())
    print(f"Rendered Figure 1 from {summary['num_interventions']} interventions")
    config_path = args.output_dir.resolve() / "resolved_config.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        settings = config.get("accuracy_analysis", {})
        if bool(settings.get("enabled", False)):
            _, report = analyze_accuracy(
                args.output_dir.resolve(),
                bootstrap_replicates=int(settings.get("bootstrap_replicates", 2000)),
                bootstrap_seed=int(settings.get("bootstrap_seed", 314159)),
                zero_tolerance=summary.get("zero_tolerance"),
            )
            print("Rendered future-KL/accuracy figure\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
