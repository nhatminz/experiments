#!/usr/bin/env python3
"""Same-g stratification for the causal state-intervention measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def matched_g_analysis(
    events: list[dict[str, Any]], requested_bins: int
) -> list[dict[str, Any]]:
    """Rank-quantile g bins; stable low/high-D split within each bin."""
    if not events:
        return []
    ordered = sorted(events, key=lambda row: (float(row["g_t"]), int(row["event_id"])))
    # A bin needs at least two events to compare low/high D. Smoke runs with two
    # events therefore produce one valid bin instead of an empty analysis.
    num_bins = max(1, min(int(requested_bins), len(ordered) // 2 or 1))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(num_bins)]
    for rank, row in enumerate(ordered):
        bins[min(num_bins - 1, rank * num_bins // len(ordered))].append(row)

    results: list[dict[str, Any]] = []
    for bin_index, rows in enumerate(bins):
        by_d = sorted(
            rows,
            key=lambda row: (float(row["sequential_gain_raw"]), int(row["event_id"])),
        )
        split = max(1, len(by_d) // 2)
        low = by_d[:split]
        high = by_d[split:]
        if not high:
            high = by_d[-1:]
            low = by_d[:1]
        low_future = _mean(low, "future_improvement")
        high_future = _mean(high, "future_improvement")
        results.append(
            {
                "g_bin": bin_index,
                "count": len(rows),
                "low_d_count": len(low),
                "high_d_count": len(high),
                "g_t_mean": _mean(rows, "g_t"),
                "g_t_min": min(float(row["g_t"]) for row in rows),
                "g_t_max": max(float(row["g_t"]) for row in rows),
                "low_d_mean": _mean(low, "sequential_gain_raw"),
                "high_d_mean": _mean(high, "sequential_gain_raw"),
                "low_d_realized_local_improvement_mean": _mean(
                    low, "realized_local_improvement"
                ),
                "high_d_realized_local_improvement_mean": _mean(
                    high, "realized_local_improvement"
                ),
                "low_d_future_improvement_mean": low_future,
                "high_d_future_improvement_mean": high_future,
                "high_minus_low_downstream_gap": high_future - low_future,
            }
        )
    return results


def analyze_output(output_dir: Path, bins: int) -> dict[str, Any]:
    events_path = output_dir / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = matched_g_analysis(events, bins)
    csv_path = output_dir / "matched_g_bins.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    summary = {
        "question": (
            "After training on a state s_t, does the distribution/quality of "
            "subsequent states visited by the student change, even when "
            "immediate gains are matched?"
        ),
        "num_interventions": len(events),
        "requested_g_bins": int(bins),
        "realized_g_bins": len(rows),
        "overall": (
            {
                "g_t_mean": _mean(events, "g_t"),
                "realized_local_improvement_mean": _mean(
                    events, "realized_local_improvement"
                ),
                "future_improvement_mean": _mean(events, "future_improvement"),
            }
            if events
            else {}
        ),
        "matched_g_bins": rows,
        "interpretation": (
            "Measurements only; no success threshold or causal conclusion is "
            "hard-coded. Compare high_minus_low_downstream_gap while checking "
            "the reported realized local improvements within each g bin."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()
    analyze_output(args.output_dir.resolve(), args.bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
