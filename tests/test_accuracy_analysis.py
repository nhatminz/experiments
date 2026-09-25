from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from Experiment.accuracy_analysis import (
    analyze_accuracy,
    cluster_resamples,
    intervention_accuracy_fields,
    validate_rollout_alignment,
)
from Experiment.plotting import summarize


def test_conditioned_accuracy_delta_and_pass_indicator():
    result = intervention_accuracy_fields(
        [True, False, False, False], [True, True, False, False]
    )
    assert result["accuracy_before"] == 0.25
    assert result["accuracy_after"] == 0.5
    assert result["delta_accuracy"] == 0.25
    assert result["pass8_before"] == result["pass8_after"] == 1


def test_cluster_bootstrap_samples_intervention_ids():
    samples = cluster_resamples([0, 0, 0, 1, 1, 2], replicates=20, seed=7)
    assert len(samples) == 20
    assert all(len(sample) == 3 for sample in samples)
    assert all(set(sample).issubset({0, 1, 2}) for sample in samples)


def test_rollout_alignment_rejects_missing_branch_record():
    events = [{"intervention_id": 0, "K": 2}]
    incomplete = [
        {"intervention_id": 0, "branch": "before", "rollout_index": index}
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="misalignment"):
        validate_rollout_alignment(events, incomplete)


def test_synthetic_analysis_writes_all_publication_outputs(tmp_path: Path):
    events = []
    continuations = []
    for event_id in range(3):
        before = [event_id == 0, False]
        after = [event_id < 2, event_id == 0]
        fields = intervention_accuracy_fields(before, after)
        delta_future = -0.02 + event_id * 0.02
        events.append({
            "intervention_id": event_id,
            "K": 2,
            "delta_immediate": 0.01 + event_id * 0.001,
            "delta_future": delta_future,
            "future_before_mean": 0.5,
            "future_after_mean": 0.5 - delta_future,
            "generated_length_before_mean": 100.0,
            "generated_length_after_mean": 110.0,
            **fields,
        })
        for branch, correctness in (("before", before), ("after", after)):
            for rollout_index, correct in enumerate(correctness):
                continuations.append({
                    "intervention_id": event_id,
                    "branch": branch,
                    "rollout_index": rollout_index,
                    "future_kl_mean": 0.1 + 0.03 * event_id + 0.01 * rollout_index,
                    "predicted_correct": correct,
                    "generated_length": 90 + 10 * rollout_index,
                    "terminated_by_eos": True,
                    "truncated_at_accuracy_limit": False,
                })
    (tmp_path / "interventions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    with gzip.open(tmp_path / "continuations.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in continuations:
            handle.write(json.dumps(row) + "\n")

    summary, report = analyze_accuracy(
        tmp_path, bootstrap_replicates=20, bootstrap_seed=3
    )
    assert summary["num_interventions"] == 3
    assert summary["bootstrap_unit"] == "intervention_id"
    assert "A. Rollout-level" in report
    for name in (
        "future_kl_accuracy_summary.json",
        "future_kl_accuracy.png",
        "future_kl_accuracy.pdf",
        "future_kl_accuracy_table.csv",
        "future_kl_accuracy_caption.txt",
        "future_kl_accuracy_analysis.tex",
    ):
        assert (tmp_path / name).is_file()


def test_runtime_uses_one_generated_rollout_for_bounded_kl_and_grading():
    source = Path(__file__).resolve().parents[1].joinpath("run_experiment.py").read_text()
    assert "kl_rollout = _slice_rollout_for_kl(rollout, future_horizon)" in source
    assert "generated_ids = [" in source
    assert "grade_evaluation_response(" in source
    assert '"num_future_states_scored": len(values)' in source


def test_accuracy_fields_do_not_change_original_figure1_statistics():
    original = [
        {
            "intervention_id": index,
            "local_reverse_kl_before": 0.5,
            "local_reverse_kl_after": 0.49,
            "delta_immediate": 0.01 + index / 1000,
            "future_quality_before_mean": 0.4,
            "future_quality_after_mean": 0.39 + index / 10000,
            "delta_future": 0.01 - index / 10000,
            "base_restoration_verified_exact": True,
        }
        for index in range(20)
    ]
    augmented = [
        dict(row, accuracy_before=0.25, accuracy_after=0.375, delta_accuracy=0.125)
        for row in original
    ]
    assert summarize(original) == summarize(augmented)
