"""Future-state KL versus Competition-MATH accuracy analysis."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_continuations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def intervention_accuracy_fields(
    before_correctness: list[bool], after_correctness: list[bool]
) -> dict[str, float | int]:
    if not before_correctness or len(before_correctness) != len(after_correctness):
        raise ValueError("Before/after correctness lists must have equal positive K")
    k = len(before_correctness)
    before_count = sum(map(int, before_correctness))
    after_count = sum(map(int, after_correctness))
    before = before_count / k
    after = after_count / k
    pass_before, pass_after = int(before_count > 0), int(after_count > 0)
    return {
        "accuracy_before": before,
        "accuracy_after": after,
        "delta_accuracy": after - before,
        "pass8_before": pass_before,
        "pass8_after": pass_after,
        "delta_pass8": pass_after - pass_before,
        "num_correct_before": before_count,
        "num_correct_after": after_count,
    }


def validate_rollout_alignment(
    interventions: list[dict[str, Any]], continuations: list[dict[str, Any]]
) -> None:
    expected = set()
    for event in interventions:
        event_id = int(event["intervention_id"])
        for branch in ("before", "after"):
            for rollout_index in range(int(event["K"])):
                expected.add((event_id, branch, rollout_index))
    actual = [
        (int(row["intervention_id"]), str(row["branch"]), int(row["rollout_index"]))
        for row in continuations
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate continuation alignment keys")
    if set(actual) != expected:
        missing = sorted(expected - set(actual))[:5]
        extra = sorted(set(actual) - expected)[:5]
        raise ValueError(f"Continuation/intervention misalignment; missing={missing}, extra={extra}")


def cluster_resamples(
    cluster_ids: list[int], replicates: int, seed: int
) -> list[list[int]]:
    """Sample clusters, never individual rollouts, with replacement."""
    unique = np.asarray(sorted(set(map(int, cluster_ids))), dtype=int)
    if unique.size == 0:
        return []
    rng = np.random.default_rng(int(seed))
    return [
        rng.choice(unique, size=unique.size, replace=True).astype(int).tolist()
        for _ in range(int(replicates))
    ]


def _expand_clusters(
    rows_by_cluster: dict[int, list[dict[str, Any]]], sampled: list[int]
) -> list[dict[str, Any]]:
    return [row for cluster in sampled for row in rows_by_cluster[int(cluster)]]


def _ci(values: list[float]) -> list[float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return [float(x) for x in np.quantile(finite, [0.025, 0.975])]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2.0
        begin = end
    return ranks


def _correlation(x: list[float], y: list[float], *, spearman: bool = False) -> float:
    if len(x) < 2:
        return float("nan")
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if spearman:
        a, b = _rank(a), _rank(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _quintile_groups(
    rows: list[dict[str, Any]], value_key: str
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row[value_key]), int(row.get("intervention_id", 0))),
    )
    return [
        [ordered[int(index)] for index in indices]
        for indices in np.array_split(np.arange(len(ordered)), 5)
    ]


def _rollout_quintiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, group in enumerate(_quintile_groups(rows, "future_kl_mean"), start=1):
        result.append({
            "quintile": index,
            "label": "Low KL" if index == 1 else ("High KL" if index == 5 else f"Q{index}"),
            "count": len(group),
            "mean_future_kl": (
                fmean(float(row["future_kl_mean"]) for row in group) if group else None
            ),
            "accuracy": (
                fmean(float(bool(row["predicted_correct"])) for row in group)
                if group else None
            ),
        })
    return result


def rollout_level_analysis(
    rows: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("future_kl_mean") is not None]
    quintiles = _rollout_quintiles(valid)
    x = [float(row["future_kl_mean"]) for row in valid]
    y = [float(bool(row["predicted_correct"])) for row in valid]
    point = _correlation(x, y)
    rows_by_cluster: dict[int, list[dict[str, Any]]] = {}
    for row in valid:
        rows_by_cluster.setdefault(int(row["intervention_id"]), []).append(row)
    bootstrap_quintiles: list[list[float]] = [[] for _ in range(5)]
    bootstrap_differences: list[float] = []
    bootstrap_correlations: list[float] = []
    for sampled in cluster_resamples(list(rows_by_cluster), replicates, seed):
        sample = _expand_clusters(rows_by_cluster, sampled)
        bins = _rollout_quintiles(sample)
        accuracies = [item["accuracy"] for item in bins]
        for index, value in enumerate(accuracies):
            if value is not None:
                bootstrap_quintiles[index].append(float(value))
        if accuracies[0] is not None and accuracies[-1] is not None:
            bootstrap_differences.append(float(accuracies[0] - accuracies[-1]))
        bootstrap_correlations.append(_correlation(
            [float(row["future_kl_mean"]) for row in sample],
            [float(bool(row["predicted_correct"])) for row in sample],
        ))
    for index, item in enumerate(quintiles):
        item["accuracy_ci95"] = _ci(bootstrap_quintiles[index])
    q1 = quintiles[0]["accuracy"]
    q5 = quintiles[-1]["accuracy"]
    payload: dict[str, Any] = {
        "num_rollouts": len(rows),
        "num_valid_kl_rollouts": len(valid),
        "excluded_zero_descendant_states": len(rows) - len(valid),
        "quintiles": quintiles,
        "q1_accuracy": q1,
        "q5_accuracy": q5,
        "q1_minus_q5": (
            float(q1 - q5) if q1 is not None and q5 is not None else None
        ),
        "q1_minus_q5_ci95": _ci(bootstrap_differences),
        "point_biserial": None if not math.isfinite(point) else point,
        "point_biserial_ci95": _ci(bootstrap_correlations),
    }
    try:
        from sklearn.metrics import roc_auc_score

        payload["auroc_using_negative_future_kl"] = (
            float(roc_auc_score(y, [-value for value in x]))
            if len(set(y)) > 1 else None
        )
    except (ImportError, ValueError):
        payload["auroc_using_negative_future_kl"] = None
    return payload


def _intervention_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = _quintile_groups(rows, "delta_future")
    return [
        {
            "quintile": index,
            "count": len(group),
            "mean_delta_future": (
                fmean(float(row["delta_future"]) for row in group) if group else None
            ),
            "mean_delta_accuracy": (
                fmean(float(row["delta_accuracy"]) for row in group) if group else None
            ),
        }
        for index, group in enumerate(bins, start=1)
    ]


def _change_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["delta_accuracy"]) for row in rows]
    return {
        "count": len(rows),
        "mean_delta_accuracy": fmean(values) if values else None,
        "median_delta_accuracy": median(values) if values else None,
        "fraction_accuracy_positive": (
            sum(value > 0 for value in values) / len(values) if values else None
        ),
        "fraction_accuracy_negative": (
            sum(value < 0 for value in values) / len(values) if values else None
        ),
    }


def intervention_level_analysis(
    rows: list[dict[str, Any]], replicates: int, seed: int, zero_tolerance: float
) -> dict[str, Any]:
    x = [float(row["delta_future"]) for row in rows]
    y = [float(row["delta_accuracy"]) for row in rows]
    pearson, spearman = _correlation(x, y), _correlation(x, y, spearman=True)
    pearsons, spearmans = [], []
    bin_bootstrap: list[list[float]] = [[] for _ in range(5)]
    rng = np.random.default_rng(int(seed))
    for _ in range(int(replicates)):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(index)] for index in indices]
        sx = [float(row["delta_future"]) for row in sample]
        sy = [float(row["delta_accuracy"]) for row in sample]
        pearsons.append(_correlation(sx, sy))
        spearmans.append(_correlation(sx, sy, spearman=True))
        for index, item in enumerate(_intervention_bins(sample)):
            if item["mean_delta_accuracy"] is not None:
                bin_bootstrap[index].append(float(item["mean_delta_accuracy"]))
    bins = _intervention_bins(rows)
    for index, item in enumerate(bins):
        item["mean_delta_accuracy_ci95"] = _ci(bin_bootstrap[index])
    positive = [row for row in rows if float(row["delta_future"]) > zero_tolerance]
    negative = [row for row in rows if float(row["delta_future"]) < -zero_tolerance]
    near = [row for row in rows if abs(float(row["delta_future"])) <= zero_tolerance]
    return {
        "pearson_delta_future_delta_accuracy": (
            pearson if math.isfinite(pearson) else None
        ),
        "pearson_ci95": _ci(pearsons),
        "spearman_delta_future_delta_accuracy": (
            spearman if math.isfinite(spearman) else None
        ),
        "spearman_ci95": _ci(spearmans),
        "zero_tolerance": float(zero_tolerance),
        "delta_future_quantile_bins": bins,
        "positive_future_change_group": _change_group(positive),
        "negative_future_change_group": _change_group(negative),
        "near_zero_group": _change_group(near),
    }


def _length_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("future_kl_mean") is not None]
    correlation = _correlation(
        [float(row["future_kl_mean"]) for row in valid],
        [float(row["generated_length"]) for row in valid],
    )
    groups = _quintile_groups(valid, "generated_length")
    return {
        "future_kl_generated_length_correlation": (
            correlation if math.isfinite(correlation) else None
        ),
        "accuracy_by_generation_length_quintile": [
            {
                "quintile": index,
                "count": len(group),
                "mean_length": fmean(float(row["generated_length"]) for row in group) if group else None,
                "accuracy": fmean(float(bool(row["predicted_correct"])) for row in group) if group else None,
            }
            for index, group in enumerate(groups, start=1)
        ],
    }


def _write_table(rows: list[dict[str, Any]], output: Path) -> None:
    fields = (
        "intervention_id", "delta_immediate", "delta_future",
        "accuracy_before", "accuracy_after", "delta_accuracy",
        "pass8_before", "pass8_after", "delta_pass8",
        "future_before_mean", "future_after_mean",
        "generated_length_before_mean", "generated_length_after_mean",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _render_figure(summary: dict[str, Any], interventions, output_dir: Path) -> None:
    quintiles = summary["rollout_level"]["all"]["quintiles"]
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    ax = axes[0]
    qx = np.arange(1, 6)
    qy = np.asarray([100.0 * float(item["accuracy"]) for item in quintiles])
    lower = np.asarray([
        100.0 * (float(item["accuracy"]) - float(item["accuracy_ci95"][0]))
        if item["accuracy_ci95"] else 0.0 for item in quintiles
    ])
    upper = np.asarray([
        100.0 * (float(item["accuracy_ci95"][1]) - float(item["accuracy"]))
        if item["accuracy_ci95"] else 0.0 for item in quintiles
    ])
    ax.errorbar(qx, qy, yerr=np.vstack((lower, upper)), marker="o", linewidth=1.3,
                capsize=3, color="#287f7b", ecolor="#7aa6a3")
    ax.set_xticks(qx, ["Q1\nLow KL", "Q2", "Q3", "Q4", "Q5\nHigh KL"])
    ax.set_xlabel("Future KL quintile (low → high)")
    ax.set_ylabel("Continuation accuracy (%)")
    ax.set_title("(a) Accuracy vs. future KL", loc="left", fontweight="bold")

    ax = axes[1]
    x = np.asarray([float(row["delta_future"]) for row in interventions])
    y = np.asarray([float(row["delta_accuracy"]) for row in interventions])
    jitter = np.random.default_rng(0).uniform(-0.012, 0.012, size=len(y))
    ax.scatter(x, y + jitter, s=19, alpha=0.38, color="#607d9b", edgecolors="none")
    bins = [
        item
        for item in summary["intervention_level"]["delta_future_quantile_bins"]
        if item["mean_delta_future"] is not None
        and item["mean_delta_accuracy"] is not None
    ]
    bx = np.asarray([float(item["mean_delta_future"]) for item in bins])
    by = np.asarray([float(item["mean_delta_accuracy"]) for item in bins])
    blo = np.asarray([
        float(item["mean_delta_accuracy"]) - float(item["mean_delta_accuracy_ci95"][0])
        if item["mean_delta_accuracy_ci95"] else 0.0 for item in bins
    ])
    bhi = np.asarray([
        float(item["mean_delta_accuracy_ci95"][1]) - float(item["mean_delta_accuracy"])
        if item["mean_delta_accuracy_ci95"] else 0.0 for item in bins
    ])
    ax.errorbar(bx, by, yerr=np.vstack((blo, bhi)), marker="o", linewidth=1.25,
                capsize=3, color="#d17a2b", ecolor="#d7a77d", zorder=3)
    ax.axvline(0.0, color="0.3", linestyle="--", linewidth=0.9)
    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Downstream KL change (before − after)")
    ax.set_ylabel("Accuracy change (after − before)")
    ax.set_title("(b) KL reduction vs. accuracy gain", loc="left", fontweight="bold")
    figure.savefig(output_dir / "future_kl_accuracy.png", dpi=320, bbox_inches="tight")
    figure.savefig(output_dir / "future_kl_accuracy.pdf", bbox_inches="tight")
    plt.close(figure)


def _caption(summary: dict[str, Any]) -> str:
    rollout = summary["rollout_level"]["all"]
    intervention = summary["intervention_level"]
    q_ci_pp = (
        [100.0 * value for value in rollout["q1_minus_q5_ci95"]]
        if rollout["q1_minus_q5_ci95"] else None
    )
    text = (
        "Lower future-state KL and task success. (a) Competition-MATH continuation "
        "accuracy across future-KL quintiles; error bars are 95% confidence intervals "
        "from intervention-clustered bootstrap. (b) Change in future-state KL versus "
        "change in conditioned accuracy across independent single-state interventions. "
        "Positive downstream KL change denotes lower future KL after intervention; "
        "positive accuracy change denotes improved solution accuracy. "
        f"Q1−Q5 accuracy is {100.0 * rollout['q1_minus_q5']:.2f} percentage points "
        f"(95% CI {q_ci_pp} percentage points); Pearson association between changes "
        f"is {intervention['pearson_delta_future_delta_accuracy']} "
        f"(95% CI {intervention['pearson_ci95']})."
    )
    accuracies = [item["accuracy"] for item in rollout["quintiles"]]
    if all(a is not None for a in accuracies) and all(
        accuracies[index] >= accuracies[index + 1] for index in range(4)
    ):
        text += " Accuracy is non-increasing across the measured KL quintiles."
    return text


def _latex(summary: dict[str, Any]) -> str:
    all_rows = summary["rollout_level"]["all"]
    before = summary["rollout_level"]["before_only"]
    intervention = summary["intervention_level"]
    all_ci = [100.0 * value for value in all_rows["q1_minus_q5_ci95"]]
    before_ci = [100.0 * value for value in before["q1_minus_q5_ci95"]]
    return (
        "We test whether the fixed-evaluator future-state KL used in our mechanistic "
        "analysis is associated with task success, without treating this association "
        "as causal. Across all sampled continuations, the lowest-minus-highest KL "
        f"quintile accuracy difference is {100 * all_rows['q1_minus_q5']:.2f} percentage "
        f"points (cluster-bootstrap 95\\% CI {all_ci}). "
        f"For untouched base-model continuations alone, the difference is "
        f"{100 * before['q1_minus_q5']:.2f} points (95\\% CI "
        f"{before_ci}). At intervention level, the Pearson "
        "correlation between downstream KL reduction and conditioned accuracy gain is "
        f"{intervention['pearson_delta_future_delta_accuracy']} (95\\% CI "
        f"{intervention['pearson_ci95']}); the corresponding Spearman correlation is "
        f"{intervention['spearman_delta_future_delta_accuracy']} (95\\% CI "
        f"{intervention['spearman_ci95']}). These measurements establish association "
        "at this checkpoint and sampling protocol, not causal sufficiency of KL.\n"
    )


def analyze_accuracy(
    output_dir: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    zero_tolerance: float | None = None,
) -> tuple[dict[str, Any], str]:
    interventions = [
        json.loads(line)
        for line in (output_dir / "interventions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    continuations = read_continuations(output_dir / "continuations.jsonl.gz")
    if not interventions or not continuations:
        raise ValueError("Accuracy analysis requires intervention and continuation records")
    validate_rollout_alignment(interventions, continuations)
    if zero_tolerance is None:
        values = np.asarray([float(row["delta_future"]) for row in interventions])
        zero_tolerance = max(1.0e-8, 0.05 * float(np.std(values)))
    rollout_all = rollout_level_analysis(continuations, bootstrap_replicates, bootstrap_seed)
    before = rollout_level_analysis(
        [row for row in continuations if row["branch"] == "before"],
        bootstrap_replicates, bootstrap_seed + 1,
    )
    after = rollout_level_analysis(
        [row for row in continuations if row["branch"] == "after"],
        bootstrap_replicates, bootstrap_seed + 2,
    )
    eos_rate = fmean(float(bool(row["terminated_by_eos"])) for row in continuations)
    truncation_rate = fmean(float(bool(row["truncated_at_accuracy_limit"])) for row in continuations)
    num_valid_kl = sum(
        row.get("future_kl_mean") is not None for row in continuations
    )
    num_excluded_zero_kl = len(continuations) - num_valid_kl
    summary = {
        "num_interventions": len(interventions),
        "num_rollouts": len(continuations),
        "num_valid_kl_rollouts": num_valid_kl,
        "num_excluded_zero_descendant_rollouts": num_excluded_zero_kl,
        "kl_exclusion_reason": "zero valid descendant states",
        "eos_rate": eos_rate,
        "truncation_rate": truncation_rate,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_unit": "intervention_id",
        "rollout_level": {
            "all": rollout_all,
            "before_only": before,
            "after_only": after,
            "quintiles": rollout_all["quintiles"],
            "q1_accuracy": rollout_all["q1_accuracy"],
            "q5_accuracy": rollout_all["q5_accuracy"],
            "q1_minus_q5": rollout_all["q1_minus_q5"],
            "q1_minus_q5_ci95": rollout_all["q1_minus_q5_ci95"],
            "point_biserial": rollout_all["point_biserial"],
            "point_biserial_ci95": rollout_all["point_biserial_ci95"],
        },
        "intervention_level": intervention_level_analysis(
            interventions, bootstrap_replicates, bootstrap_seed + 3, float(zero_tolerance)
        ),
        "continuation_length_check": _length_analysis(continuations),
        "interpretation": "Associational diagnostic only; no causal claim is made.",
    }
    (output_dir / "future_kl_accuracy_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_table(interventions, output_dir / "future_kl_accuracy_table.csv")
    _render_figure(summary, interventions, output_dir)
    (output_dir / "future_kl_accuracy_caption.txt").write_text(
        _caption(summary) + "\n", encoding="utf-8"
    )
    (output_dir / "future_kl_accuracy_analysis.tex").write_text(
        _latex(summary), encoding="utf-8"
    )
    rollout_supported = bool(
        rollout_all["q1_minus_q5"] is not None
        and rollout_all["q1_minus_q5"] > 0
        and rollout_all["q1_minus_q5_ci95"]
        and rollout_all["q1_minus_q5_ci95"][0] > 0
    )
    before_supported = bool(
        before["q1_minus_q5"] is not None
        and before["q1_minus_q5"] > 0
        and before["q1_minus_q5_ci95"]
        and before["q1_minus_q5_ci95"][0] > 0
    )
    change_ci = summary["intervention_level"]["pearson_ci95"]
    change_supported = bool(
        summary["intervention_level"]["pearson_delta_future_delta_accuracy"] is not None
        and summary["intervention_level"]["pearson_delta_future_delta_accuracy"] > 0
        and change_ci and change_ci[0] > 0
    )
    report = (
        "A. Rollout-level: "
        + ("supported association; " if rollout_supported else "not conclusively supported; ")
        + "Q1-Q5 accuracy difference="
        f"{100 * rollout_all['q1_minus_q5']:.2f} pp, CI="
        f"{[100 * value for value in rollout_all['q1_minus_q5_ci95']]} pp; "
        f"point-biserial={rollout_all['point_biserial']}, CI={rollout_all['point_biserial_ci95']}.\n"
        "B. Before-only: "
        + ("supported association; " if before_supported else "not conclusively supported; ")
        + "Q1-Q5 accuracy difference="
        f"{100 * before['q1_minus_q5']:.2f} pp, CI="
        f"{[100 * value for value in before['q1_minus_q5_ci95']]} pp.\n"
        "C. Intervention-level: "
        + ("supported positive association; " if change_supported else "not conclusively supported; ")
        + "Pearson(delta_future, delta_accuracy)="
        f"{summary['intervention_level']['pearson_delta_future_delta_accuracy']}, "
        f"CI={summary['intervention_level']['pearson_ci95']}; Spearman="
        f"{summary['intervention_level']['spearman_delta_future_delta_accuracy']}, "
        f"CI={summary['intervention_level']['spearman_ci95']}.\n"
        f"D. Truncation: EOS={100*eos_rate:.2f}%, truncated={100*truncation_rate:.2f}%, "
        f"excluded from rollout-level KL analysis={num_excluded_zero_kl} "
        "(zero valid descendant states); "
        + ("generation limit appears adequate." if truncation_rate <= 0.02 else "generation limit may be inadequate.")
        + " These are associations, not causal conclusions."
    )
    return summary, report
