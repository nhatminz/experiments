"""Analysis and publication-quality Figure 1 rendering."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml



def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return result


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def matched_band(
    rows: list[dict[str, Any]], low_q: float = 0.45, high_q: float = 0.55
) -> tuple[list[dict[str, Any]], tuple[float, float]]:
    if not rows:
        return [], (float("nan"), float("nan"))
    values = np.asarray([float(row["delta_immediate"]) for row in rows])
    lower, upper = np.quantile(values, [low_q, high_q])
    selected = [
        row for row in rows if lower <= float(row["delta_immediate"]) <= upper
    ]
    return selected, (float(lower), float(upper))


def summarize(
    rows: list[dict[str, Any]],
    low_q: float = 0.45,
    high_q: float = 0.55,
    sensitivity_low_q: float = 0.40,
    sensitivity_high_q: float = 0.60,
) -> dict[str, Any]:
    if not rows:
        return {"num_interventions": 0}
    immediate = np.asarray([float(row["delta_immediate"]) for row in rows])
    future = np.asarray([float(row["delta_future"]) for row in rows])
    canonical, canonical_bounds = matched_band(rows, low_q, high_q)
    wider, wider_bounds = matched_band(rows, sensitivity_low_q, sensitivity_high_q)
    future_scale = float(np.std(future))
    zero_tolerance = max(1.0e-8, 0.05 * future_scale)

    def band_payload(
        selected: list[dict[str, Any]], bounds: tuple[float, float]
    ) -> dict[str, Any]:
        values = [float(row["delta_future"]) for row in selected]
        local = [float(row["delta_immediate"]) for row in selected]
        count = len(values)
        return {
            "quantile_bounds": list(bounds),
            "count": len(selected),
            "delta_immediate_mean": fmean(local) if local else None,
            "delta_immediate_std": float(np.std(local)) if local else None,
            "delta_immediate_min": min(local) if local else None,
            "delta_immediate_max": max(local) if local else None,
            "delta_future_mean": fmean(values) if values else None,
            "delta_future_std": float(np.std(values)) if values else None,
            "delta_future_min": min(values) if values else None,
            "delta_future_max": max(values) if values else None,
            "positive_count": sum(value > zero_tolerance for value in values),
            "positive_percentage": 100.0 * sum(value > zero_tolerance for value in values) / max(count, 1),
            "negative_count": sum(value < -zero_tolerance for value in values),
            "negative_percentage": 100.0 * sum(value < -zero_tolerance for value in values) / max(count, 1),
            "approximately_zero_count": sum(
                abs(value) <= zero_tolerance for value in values
            ),
            "approximately_zero_percentage": 100.0 * sum(
                abs(value) <= zero_tolerance for value in values
            ) / max(count, 1),
            "both_signs_observed": (
                any(value > zero_tolerance for value in values)
                and any(value < -zero_tolerance for value in values)
            ),
        }

    return {
        "question": (
            "After one teacher-supervised update at a student-visited state, "
            "does the distribution of later visited-state quality change when "
            "realized immediate gain is matched?"
        ),
        "num_interventions": len(rows),
        "num_continuations_per_branch": rows[0].get("K"),
        "delta_immediate": {
            "mean": float(np.mean(immediate)),
            "std": float(np.std(immediate)),
            "min": float(np.min(immediate)),
            "max": float(np.max(immediate)),
        },
        "delta_future": {
            "mean": float(np.mean(future)),
            "std": future_scale,
            "min": float(np.min(future)),
            "max": float(np.max(future)),
        },
        "zero_tolerance": zero_tolerance,
        "pearson": _correlation(immediate, future),
        "spearman": _correlation(_ranks(immediate), _ranks(future)),
        "matched_q45_q55": band_payload(canonical, canonical_bounds),
        "sensitivity_q40_q60": band_payload(wider, wider_bounds),
        "matched_quantiles": [low_q, high_q],
        "sensitivity_quantiles": [sensitivity_low_q, sensitivity_high_q],
        "claim_boundary": (
            "This is a checkpoint-local categorical intervention diagnostic. "
            "It does not identify the full neural-parameter causal effect of a "
            "training token and uses no CMT/LIFT score."
        ),
    }


def _caption(summary: dict[str, Any]) -> str:
    band = summary["matched_q45_q55"]
    low_q, high_q = summary["matched_quantiles"]
    band_name = f"Q{100 * low_q:g}--Q{100 * high_q:g}"
    base = (
        "Figure 1: Similar immediate learning gains can lead to different "
        "downstream supervision. Using Qwen3-4B to Qwen3-1.7B-Base on "
        "Competition-MATH, each point is an independent branch from the same "
        "base checkpoint. A single AdamW step minimizes local reverse KL on the "
        "fixed Student Top-16 support; both suffix sets are then evaluated by "
        "the restored base student and frozen teacher. "
        f"Across N={summary['num_interventions']} interventions"
        + (
            f" with K={summary['num_continuations_per_branch']} continuations per branch"
            if summary.get("num_continuations_per_branch") is not None else ""
        )
        + f", the {band_name} "
        f"immediate-gain band [{band['quantile_bounds'][0]:.4g}, "
        f"{band['quantile_bounds'][1]:.4g}] contains {band['count']} interventions"
    )
    if band["both_signs_observed"]:
        base += (
            f", including {band['positive_count']} positive and "
            f"{band['negative_count']} negative downstream changes"
        )
    else:
        base += "; the matched band does not contain robust evidence of both signs"
    return base + ". Positive Δfuture means lower future reverse KL after intervention."


def render_figure(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    low_q: float = 0.45,
    high_q: float = 0.55,
    sensitivity_low_q: float = 0.40,
    sensitivity_high_q: float = 0.60,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot plot an empty intervention table")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, low_q, high_q, sensitivity_low_q, sensitivity_high_q)
    selected, bounds = matched_band(rows, low_q, high_q)
    band_name = f"Q{100 * low_q:g}–Q{100 * high_q:g} Δimmediate"
    immediate = np.asarray([float(row["delta_immediate"]) for row in rows])
    future = np.asarray([float(row["delta_future"]) for row in rows])

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    ax = axes[0]
    inside = (immediate >= bounds[0]) & (immediate <= bounds[1])
    ax.scatter(immediate[~inside], future[~inside], s=22, alpha=0.58, color="#607d9b", edgecolors="none")
    ax.scatter(immediate[inside], future[inside], s=34, alpha=0.9, color="#2a9d8f", edgecolors="white", linewidths=0.35, zorder=3)
    ax.axhline(0.0, color="0.25", linewidth=1.0, linestyle="--")
    ax.axvspan(bounds[0], bounds[1], color="#72b7b2", alpha=0.18, label=band_name)
    ax.set_xlabel("Realized immediate gain, Δimmediate")
    ax.set_ylabel("Future distribution improvement, Δfuture")
    ax.set_title("(a) Immediate vs. downstream gain", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sorted_future = sorted(float(row["delta_future"]) for row in selected)
    colors = ["#e1812c" if value < 0 else "#2a9d8f" for value in sorted_future]
    x = np.arange(len(sorted_future))
    if sorted_future:
        ax.plot(x, sorted_future, color="0.62", linewidth=0.75, zorder=1)
        ax.scatter(x, sorted_future, c=colors, s=31, edgecolors="none", zorder=2)
    ax.axhline(0.0, color="0.25", linewidth=1.0)
    ax.set_xlabel("Matched interventions, sorted by Δfuture")
    ax.set_ylabel("Δfuture")
    ax.set_title("(b) Matched immediate-gain band", loc="left", fontweight="bold")
    if not sorted_future:
        ax.text(0.5, 0.5, "No Q45–Q55 observations", ha="center", va="center", transform=ax.transAxes)

    png = output_dir / "figure1.png"
    pdf = output_dir / "figure1.pdf"
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    caption = _caption(summary)
    (output_dir / "figure1_caption.txt").write_text(caption + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def analyze_output(output_dir: Path) -> dict[str, Any]:
    kwargs: dict[str, float] = {}
    config_path = output_dir / "resolved_config.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        settings = config.get("intervention", {})
        kwargs = {
            "low_q": float(settings.get("matched_quantile_low", 0.45)),
            "high_q": float(settings.get("matched_quantile_high", 0.55)),
            "sensitivity_low_q": float(settings.get("sensitivity_quantile_low", 0.40)),
            "sensitivity_high_q": float(settings.get("sensitivity_quantile_high", 0.60)),
        }
    return render_figure(
        _read_jsonl(output_dir / "interventions.jsonl"), output_dir, **kwargs
    )
