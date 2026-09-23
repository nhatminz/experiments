"""Pure protocol invariants for the CMT state-intervention experiment."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def eligible_state_positions(
    response_length: int, min_original_suffix_tokens: int
) -> list[int]:
    """Return t where y_t exists and enough tokens remain strictly after it."""
    length = int(response_length)
    minimum = int(min_original_suffix_tokens)
    if length < 0 or minimum < 1:
        raise ValueError("response_length must be >=0 and minimum suffix must be >=1")
    # At state s_t, y_t is the next action. Tokens y_{t+1:} are its observed
    # original suffix, so length - t - 1 must be at least `minimum`.
    return list(range(max(0, length - minimum)))


def select_state_position(
    response_length: int,
    min_original_suffix_tokens: int,
    *,
    seed: int,
) -> int:
    """Uniform state selection independent of CMT D_t and future outcomes."""
    positions = eligible_state_positions(response_length, min_original_suffix_tokens)
    if not positions:
        raise ValueError("Trajectory has no eligible non-terminal intervention state")
    return positions[random.Random(int(seed)).randrange(len(positions))]


def build_state_prefix(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    position_t: int,
) -> torch.Tensor:
    """Construct s_t = prompt + y_<t, explicitly excluding action y_t."""
    if prompt_ids.ndim != 1 or prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("Prompt IDs and mask must be aligned 1-D tensors")
    if response_ids.ndim != 1:
        raise ValueError("Response IDs must be a 1-D tensor")
    t = int(position_t)
    if not 0 <= t < response_ids.numel():
        raise IndexError(
            f"position_t={t} is outside response length {response_ids.numel()}"
        )
    prompt = prompt_ids[prompt_attention_mask.bool()]
    prefix = torch.cat((prompt, response_ids[:t]), dim=0)
    expected_length = int(prompt.numel()) + t
    if prefix.numel() != expected_length:
        raise AssertionError("State-prefix construction violated the t/y_t boundary")
    return prefix


def single_state_objective_mask(
    valid_mask: torch.Tensor, position_t: int
) -> torch.Tensor:
    """Mask exactly one valid state and assert no other position is supervised."""
    if valid_mask.ndim != 2 or valid_mask.shape[0] != 1:
        raise ValueError("Local intervention expects one trajectory [1, time]")
    t = int(position_t)
    if not 0 <= t < valid_mask.shape[1] or not bool(valid_mask[0, t]):
        raise ValueError("Intervention position must be valid")
    mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    mask[0, t] = True
    if int(mask.sum().item()) != 1 or bool((mask & ~valid_mask.bool()).any()):
        raise AssertionError("Local intervention mask must contain exactly one state")
    return mask


def paired_rollout_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    if int(count) <= 0:
        raise ValueError("num_probe_rollouts must be positive")
    return tuple(int(base_seed) + index for index in range(int(count)))


def assert_paired_probe_protocol(
    before_prefix: torch.Tensor,
    after_prefix: torch.Tensor,
    before_seeds: Sequence[int],
    after_seeds: Sequence[int],
) -> None:
    if not torch.equal(before_prefix, after_prefix):
        raise AssertionError(
            "Pre/post probe rollouts must use the exact same s_t prefix"
        )
    if tuple(map(int, before_seeds)) != tuple(map(int, after_seeds)):
        raise AssertionError(
            "Pre/post probe rollouts must reuse identical paired seeds"
        )


def fixed_support_reverse_kl(
    student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor
) -> torch.Tensor:
    """KL(p_U || q_U) on one fixed candidate support U."""
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher log-probabilities must align")
    log_p = student_log_probs.float() - torch.logsumexp(
        student_log_probs.float(), dim=-1, keepdim=True
    )
    log_q = teacher_log_probs.float() - torch.logsumexp(
        teacher_log_probs.float(), dim=-1, keepdim=True
    )
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def assert_finite_values(value: Any, path: str = "event") -> None:
    """Reject NaN/Inf anywhere in numeric logging payloads."""
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
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite number at {path}: {value}")
