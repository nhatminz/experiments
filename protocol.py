"""Pure, model-independent invariants for the Figure-1 intervention study."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from statistics import fmean, pstdev
from typing import Any

import torch


def eligible_state_positions(
    response_token_ids: Sequence[int],
    *,
    eos_token_ids: Sequence[int],
    min_suffix_tokens: int = 1,
) -> list[int]:
    """Valid interior ``t`` for ``s_t = prompt + y_<t``."""
    minimum = int(min_suffix_tokens)
    if minimum < 1:
        raise ValueError("min_suffix_tokens must be >= 1")
    eos = {int(item) for item in eos_token_ids}
    length = len(response_token_ids)
    return [
        t
        for t in range(max(0, length - minimum))
        if int(response_token_ids[t]) not in eos
        and length - t - 1 >= minimum
    ]


def select_state_position(
    response_token_ids: Sequence[int],
    *,
    eos_token_ids: Sequence[int],
    min_suffix_tokens: int,
    seed: int,
    min_position: int = 0,
) -> int:
    positions = eligible_state_positions(
        response_token_ids,
        eos_token_ids=eos_token_ids,
        min_suffix_tokens=min_suffix_tokens,
    )
    preferred = [position for position in positions if position >= int(min_position)]
    if preferred:
        positions = preferred
    if not positions:
        raise ValueError("Trajectory has no valid interior intervention state")
    return positions[random.Random(int(seed)).randrange(len(positions))]


def build_state_prefix(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    position_t: int,
) -> torch.Tensor:
    """Return ``s_t`` and explicitly exclude action ``y_t``."""
    if prompt_ids.ndim != 1 or prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("Prompt IDs and mask must be aligned 1-D tensors")
    if response_ids.ndim != 1:
        raise ValueError("Response IDs must be one-dimensional")
    t = int(position_t)
    if not 0 <= t < response_ids.numel():
        raise IndexError(f"position_t={t} is outside the response")
    prompt = prompt_ids[prompt_attention_mask.bool()]
    result = torch.cat((prompt, response_ids[:t]), dim=0)
    if result.numel() != int(prompt.numel()) + t:
        raise AssertionError("State prefix crossed the y_t boundary")
    return result


def fixed_support_reverse_kl(
    student_values: torch.Tensor, teacher_values: torch.Tensor
) -> torch.Tensor:
    """``KL(p_bar || q_bar)`` after conditional normalization on fixed U."""
    if student_values.shape != teacher_values.shape:
        raise ValueError("Student and teacher values must have identical shapes")
    if student_values.shape[-1] < 1:
        raise ValueError("Fixed support cannot be empty")
    log_p = student_values.float() - torch.logsumexp(
        student_values.float(), dim=-1, keepdim=True
    )
    log_q = teacher_values.float() - torch.logsumexp(
        teacher_values.float(), dim=-1, keepdim=True
    )
    value = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Non-finite fixed-support reverse KL")
    return value


def branch_seed_sets(base_seed: int, count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministic but disjoint Monte-Carlo seeds for pre/post branches."""
    count = int(count)
    if count <= 0:
        raise ValueError("num_continuations must be positive")
    before = tuple(int(base_seed) + index for index in range(count))
    after = tuple(int(base_seed) + 1_000_000 + index for index in range(count))
    if set(before) & set(after):
        raise AssertionError("Before/after continuation seeds must be disjoint")
    return before, after


def assert_same_prefix_independent_branches(
    before_prefix: torch.Tensor,
    after_prefix: torch.Tensor,
    before_seeds: Sequence[int],
    after_seeds: Sequence[int],
) -> None:
    if not torch.equal(before_prefix, after_prefix):
        raise AssertionError("Before/after branches must start at the exact same s_t")
    if set(map(int, before_seeds)) & set(map(int, after_seeds)):
        raise AssertionError("Before/after branches must be generated independently")


def future_distribution_stats(branch_values: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Aggregate descendants; std is across continuation-level means."""
    nonempty = [[float(x) for x in branch] for branch in branch_values if branch]
    if not nonempty:
        raise ValueError("No descendant-state quality values were produced")
    flat = [value for branch in nonempty for value in branch]
    branch_means = [fmean(branch) for branch in nonempty]
    return {
        "mean": fmean(flat),
        "std_across_continuation_means": (
            pstdev(branch_means) if len(branch_means) > 1 else 0.0
        ),
        "min": min(flat),
        "max": max(flat),
        "num_states": len(flat),
        "num_nonempty_continuations": len(nonempty),
        "continuation_means": branch_means,
    }


def descendant_offsets(generated_length: int) -> list[int]:
    """Offsets scored by a causal LM after >=1 newly generated token.

    Offset zero is the original intervention state.  The scorer exposes states
    immediately before each generated action, so offsets ``1..L-1`` are the
    descendant states represented without an extra terminal forward pass.
    """
    return list(range(1, max(1, int(generated_length))))


def assert_finite_values(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_values(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_values(item, f"{path}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite tensor at {path}")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite number at {path}: {value}")
