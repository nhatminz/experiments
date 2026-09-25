from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Experiment.plotting import matched_band, render_figure, summarize
from Experiment.accuracy_analysis import (
    analyze_accuracy,
    cluster_resamples,
    intervention_accuracy_fields,
    validate_rollout_alignment,
)
from Experiment.protocol import (
    assert_same_prefix_independent_branches,
    branch_seed_sets,
    build_state_prefix,
    descendant_offsets,
    eligible_state_positions,
    fixed_support_reverse_kl,
    future_distribution_stats,
    kl_scoring_token_count,
    reconstruct_assistant_response,
)
from Experiment.run_experiment import (
    _assert_resume_schema,
    _single_local_update,
    _validate_completed_accuracy_records,
)
from Experiment.utils import (
    pending_candidates,
    restore_parameters_exact,
    snapshot_parameters,
    validate_intervention_schema,
)


class TinyLM(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.nn.Embedding(24, 8)
        self.projection = torch.nn.Linear(8, 24, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(24, 8, generator=generator))
            self.projection.weight.copy_(torch.randn(24, 8, generator=generator))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def tiny_config():
    return {
        "intervention": {"top_k": 16},
        "training": {
            "learning_rate": 5e-6,
            "adam_betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
    }


def test_state_prefix_excludes_y_t_and_valid_positions_exclude_eos():
    prompt = torch.tensor([0, 10, 11])
    mask = torch.tensor([0, 1, 1])
    response = torch.tensor([20, 21, 2, 0])
    prefix = build_state_prefix(prompt, mask, response, position_t=1)
    assert prefix.tolist() == [10, 11, 20]
    assert 2 not in eligible_state_positions(
        response.tolist(), eos_token_ids=[2], min_suffix_tokens=1
    )


def test_one_local_update_uses_fixed_student_top16_support():
    student, teacher = TinyLM(1), TinyLM(2)
    before = snapshot_parameters(student)
    result = _single_local_update(student, teacher, torch.tensor([1, 3, 5]), tiny_config())
    assert len(result["support_ids"]) == 16
    assert len(set(result["support_ids"])) == 16
    assert result["loss"] == pytest.approx(result["d0"], rel=1e-5)
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in student.named_parameters()
    )


def test_fixed_support_reverse_kl_and_support_is_not_reselected_post_update():
    p = torch.log(torch.tensor([[0.4, 0.1]]))
    q = torch.log(torch.tensor([[0.1, 0.4]]))
    value = fixed_support_reverse_kl(p, q)
    assert value.item() == pytest.approx(0.8 * torch.log(torch.tensor(4.0)).item() + 0.2 * torch.log(torch.tensor(0.25)).item())
    source = Path(__file__).resolve().parents[1].joinpath("run_experiment.py").read_text()
    assert "fixed_student_after = _next_token_logits(student, prefix).gather(-1, support_ids)" in source


def test_exact_parameter_restore_between_interventions():
    model = TinyLM(3)
    snapshot = snapshot_parameters(model)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    restore_parameters_exact(model, snapshot)
    assert all(torch.equal(parameter, snapshot[name]) for name, parameter in model.named_parameters())
    with torch.no_grad():
        next(model.parameters()).mul_(0)
    restore_parameters_exact(model, snapshot)
    assert all(torch.equal(parameter, snapshot[name]) for name, parameter in model.named_parameters())


def test_same_prefix_but_independent_deterministic_branch_seeds():
    before, after = branch_seed_sets(100, 8)
    assert before == branch_seed_sets(100, 8)[0]
    assert set(before).isdisjoint(after)
    prefix = torch.tensor([1, 2, 3])
    assert_same_prefix_independent_branches(prefix, prefix.clone(), before, after)
    with pytest.raises(AssertionError, match="independently"):
        assert_same_prefix_independent_branches(prefix, prefix, before, before)


def test_descendant_scoring_excludes_intervention_state():
    assert descendant_offsets(1) == []
    assert descendant_offsets(4) == [1, 2, 3]
    stats = future_distribution_stats([[1.0, 2.0], [3.0]])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["num_states"] == 3


def test_accuracy_generation_can_exceed_fixed_kl_horizon():
    assert kl_scoring_token_count(2048, 128) == 129
    assert len(descendant_offsets(kl_scoring_token_count(2048, 128))) == 128
    assert kl_scoring_token_count(40, 128) == 40


def test_reconstructed_response_is_assistant_only():
    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            names = {99: "USER_PROBLEM", 10: "assistant-prefix", 11: " answer"}
            return "".join(names[item] for item in ids)

    response = reconstruct_assistant_response(Tokenizer(), [10], [11])
    assert response == "assistant-prefix answer"
    assert "USER_PROBLEM" not in response


def test_accuracy_fields_use_same_k_before_and_after():
    result = intervention_accuracy_fields(
        [True, False, False, False], [True, True, False, False]
    )
    assert result["accuracy_before"] == 0.25
    assert result["accuracy_after"] == 0.5
    assert result["delta_accuracy"] == 0.25
    assert result["delta_pass8"] == 0


def test_future_scoring_occurs_only_after_exact_base_restore():
    source = Path(__file__).resolve().parents[1].joinpath("run_experiment.py").read_text()
    restore_index = source.index("restore_parameters_exact(student, base_snapshot)", source.index("del optimizer"))
    score_index = source.index("_score_suffixes_with_base(", restore_index)
    assert restore_index < score_index
    assert '"future_evaluator": "theta_base"' in source


def test_resume_skips_completed_records_without_modifying_them():
    candidates = [{"candidate_id": index} for index in range(4)]
    assert [row["candidate_id"] for row in pending_candidates(candidates, {0, 2})] == [1, 3]


def _synthetic_rows(count: int = 20):
    return [
        {
            "intervention_id": index,
            "local_reverse_kl_before": 0.5,
            "local_reverse_kl_after": 0.5 - index / 1000,
            "delta_immediate": index / 1000,
            "future_quality_before_mean": 0.4,
            "future_quality_after_mean": 0.4 - ((-1) ** index) * index / 2000,
            "delta_future": ((-1) ** index) * index / 2000,
            "base_restoration_verified_exact": True,
        }
        for index in range(count)
    ]


def test_matched_band_statistics_and_required_schema():
    rows = _synthetic_rows()
    for row in rows:
        validate_intervention_schema(row)
    selected, bounds = matched_band(rows)
    assert selected
    assert bounds[0] <= bounds[1]
    summary = summarize(rows)
    assert summary["num_interventions"] == len(rows)
    assert "matched_q45_q55" in summary
    assert "sensitivity_q40_q60" in summary
    exact = [dict(rows[0], intervention_id=i, delta_immediate=float(i)) for i in range(100)]
    selected, bounds = matched_band(exact, 0.45, 0.55)
    assert bounds == pytest.approx((44.55, 54.45))
    assert [row["intervention_id"] for row in selected] == list(range(45, 55))


def test_required_output_schema_and_figure_generation(tmp_path: Path):
    rows = _synthetic_rows()
    summary = render_figure(rows, tmp_path)
    assert summary["num_interventions"] == len(rows)
    for name in ("summary.json", "figure1_caption.txt", "figure1.png", "figure1.pdf"):
        assert (tmp_path / name).is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["num_interventions"] == len(rows)


def test_runtime_does_not_import_cmt_or_lift_selectors():
    source = Path(__file__).resolve().parents[1].joinpath("run_experiment.py").read_text()
    assert "CMTSelector" not in source
    assert "PGTSelector" not in source
    assert "_opd_train_step" not in source
    assert "grade_evaluation_response" in source
    assert "kl_rollout = _slice_rollout_for_kl(rollout, future_horizon)" in source
    assert "generated_ids" in source
    assert "reconstruct_assistant_response(" in source
    assert '"num_future_states_scored": len(values)' in source


def test_cluster_bootstrap_samples_interventions_not_rollouts():
    samples = cluster_resamples([0, 0, 0, 1, 1, 2], 20, seed=7)
    assert len(samples) == 20
    assert all(len(sample) == 3 for sample in samples)
    assert all(set(sample).issubset({0, 1, 2}) for sample in samples)


def test_alignment_and_schema_v2_resume_rejection():
    events = [{"intervention_id": 0, "K": 2}]
    continuations = [
        {"intervention_id": 0, "branch": branch, "rollout_index": index}
        for branch in ("before", "after") for index in range(2)
    ]
    validate_rollout_alignment(events, continuations)
    with pytest.raises(ValueError, match="schema version 2"):
        _assert_resume_schema(
            {"experiment": {}},
            {"experiment": {"experiment_schema_version": 2}},
        )
    with pytest.raises(ValueError, match="full-generation accuracy fields"):
        _validate_completed_accuracy_records(
            [{"intervention_id": 0, "experiment_schema_version": 1}],
            enabled=True,
        )


def test_accuracy_analysis_smoke_and_original_figure1_unchanged(tmp_path: Path):
    events = []
    continuations = []
    for event_id in range(3):
        before_correct = [event_id == 0, False]
        after_correct = [event_id < 2, event_id == 0]
        fields = intervention_accuracy_fields(before_correct, after_correct)
        events.append({
            "intervention_id": event_id,
            "K": 2,
            "delta_immediate": 0.01 + event_id * 0.001,
            "delta_future": -0.02 + event_id * 0.02,
            "future_before_mean": 0.5,
            "future_after_mean": 0.5 - (-0.02 + event_id * 0.02),
            "generated_length_before_mean": 100.0,
            "generated_length_after_mean": 110.0,
            **fields,
        })
        for branch, correctness in (("before", before_correct), ("after", after_correct)):
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
    import gzip
    with gzip.open(tmp_path / "continuations.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in continuations:
            handle.write(json.dumps(row) + "\n")
    original_rows = _synthetic_rows()
    augmented_rows = [dict(row, delta_accuracy=0.0) for row in original_rows]
    assert summarize(original_rows) == summarize(augmented_rows)
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
