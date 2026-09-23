from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from analyze import analyze_output, matched_g_analysis
from protocol import (
    assert_finite_values,
    assert_paired_probe_protocol,
    build_state_prefix,
    eligible_state_positions,
    paired_rollout_seeds,
    fixed_support_reverse_kl,
    single_state_objective_mask,
)
from run_experiment import _slice_local_rollout
from b200_experiment.scoring import RolloutBatch


def test_state_prefix_excludes_y_t():
    prompt = torch.tensor([0, 0, 10, 11, 12])
    mask = torch.tensor([0, 0, 1, 1, 1])
    response = torch.tensor([20, 21, 22, 23])
    prefix = build_state_prefix(prompt, mask, response, position_t=2)
    assert prefix.tolist() == [10, 11, 12, 20, 21]
    assert prefix[-1].item() != response[2].item()
    assert prefix.numel() == int(mask.sum()) + 2


def test_eligible_positions_require_suffix_strictly_after_y_t():
    assert eligible_state_positions(
        response_length=6, min_original_suffix_tokens=2
    ) == [
        0,
        1,
        2,
        3,
    ]
    assert (
        eligible_state_positions(response_length=2, min_original_suffix_tokens=2) == []
    )


def test_local_objective_has_one_state_and_other_positions_have_zero_gradient():
    valid = torch.tensor([[True, True, True, True]])
    objective = single_state_objective_mask(valid, position_t=2)
    assert objective.sum().item() == 1
    per_position_loss = torch.arange(4.0, requires_grad=True).reshape(1, 4)
    weighted = (per_position_loss * objective.float()).sum()
    gradient = torch.autograd.grad(weighted, per_position_loss)[0]
    assert gradient.tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_local_rollout_contains_y_t_as_target_but_no_descendant_tokens():
    rollout = RolloutBatch(
        input_ids=torch.tensor([[10, 11, 20, 21, 22, 23]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        response_ids=torch.tensor([[20, 21, 22, 23]]),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        rollout_log_probs=torch.zeros(1, 4),
        prompt_width=2,
    )
    local = _slice_local_rollout(rollout, position_t=2)
    assert local.response_ids.tolist() == [[20, 21, 22]]
    assert local.input_ids.tolist() == [[10, 11, 20, 21, 22]]
    objective = single_state_objective_mask(local.valid_mask, position_t=2)
    assert objective.tolist() == [[False, False, True]]


def test_fixed_support_reverse_kl_renormalizes_both_distributions():
    student = torch.log(torch.tensor([[0.4, 0.1]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))
    value = fixed_support_reverse_kl(student, teacher)
    expected = 0.8 * math.log(4.0) + 0.2 * math.log(0.25)
    assert value.item() == pytest.approx(expected)


def test_pre_post_prefix_and_paired_seeds_must_match():
    prefix = torch.tensor([1, 2, 3])
    seeds = paired_rollout_seeds(100, 4)
    assert seeds == (100, 101, 102, 103)
    assert_paired_probe_protocol(prefix, prefix.clone(), seeds, tuple(seeds))
    with pytest.raises(AssertionError, match="same s_t prefix"):
        assert_paired_probe_protocol(prefix, torch.tensor([1, 2, 4]), seeds, seeds)
    with pytest.raises(AssertionError, match="identical paired seeds"):
        assert_paired_probe_protocol(prefix, prefix, seeds, (100, 101, 102, 104))


def test_logged_values_must_be_finite():
    assert_finite_values({"x": 1.0, "nested": [2, 3.0]})
    with pytest.raises(FloatingPointError):
        assert_finite_values({"x": float("nan")})
    with pytest.raises(FloatingPointError):
        assert_finite_values({"x": torch.tensor(float("inf"))})


def _event(index: int, g: float, d: float, local: float, future: float):
    return {
        "event_id": index,
        "g_t": g,
        "sequential_gain_raw": d,
        "realized_local_improvement": local,
        "future_improvement": future,
    }


def test_matched_g_analysis_and_files_do_not_hardcode_success(tmp_path: Path):
    events = [
        _event(0, 0.10, -1.0, 0.01, -0.02),
        _event(1, 0.11, 1.0, 0.011, 0.04),
        _event(2, 0.50, -2.0, 0.02, 0.01),
        _event(3, 0.51, 2.0, 0.021, 0.05),
    ]
    bins = matched_g_analysis(events, 2)
    assert len(bins) == 2
    assert bins[0]["high_minus_low_downstream_gap"] == pytest.approx(0.06)
    with (tmp_path / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    summary = analyze_output(tmp_path, bins=2)
    assert summary["num_interventions"] == 4
    assert "verdict" not in summary
    assert '"success":' not in json.dumps(summary).lower()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "matched_g_bins.csv").is_file()
