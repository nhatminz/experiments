#!/usr/bin/env python3
"""Sequential causal interventions at student-generated CMT states."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import torch

from analyze import analyze_output
from gpu_isolation import exclusive_gpu_lease
from protocol import (
    assert_finite_values,
    assert_paired_probe_protocol,
    build_state_prefix,
    fixed_support_reverse_kl,
    paired_rollout_seeds,
    select_state_position,
    single_state_objective_mask,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _cuda_device_identity(device_index: int = 0) -> str:
    """Return a stable identity for the physical device behind logical CUDA 0."""
    properties = torch.cuda.get_device_properties(device_index)
    for attribute in ("uuid", "pci_bus_id"):
        value = getattr(properties, attribute, None)
        if value:
            return str(value)
    # Older PyTorch builds may omit UUID and PCI bus ID. The launcher enforces
    # exactly one CUDA_VISIBLE_DEVICES entry, which is still a better identity
    # than logical cuda:0 (every independently launched process calls it zero).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return f"visible-{visible}"
    return (
        f"logical-{device_index}-{properties.name}-"
        f"{int(properties.total_memory)}"
    )


def _validate_vllm_startup_memory(config: dict[str, Any]) -> dict[str, float]:
    """Fail before vLLM startup when another process already occupies the GPU."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    settings = config["rollout"]["vllm"]
    utilization = float(settings["gpu_memory_utilization"])
    headroom_gib = _env_float("VLLM_STARTUP_HEADROOM_GIB", 4.0)
    if headroom_gib < 0:
        raise ValueError("VLLM_STARTUP_HEADROOM_GIB cannot be negative")
    required_bytes = int(total_bytes * utilization + headroom_gib * 2**30)
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough free VRAM to start the Experiment vLLM replica: "
            f"need at least {required_bytes / 2**30:.1f} GiB free "
            f"(gpu_memory_utilization={utilization:.3f} plus "
            f"{headroom_gib:.1f} GiB startup headroom), but only "
            f"{free_bytes / 2**30:.1f} of {total_bytes / 2**30:.1f} GiB is free. "
            "This normally means another process is using the selected physical "
            "GPU. Do not bypass this check by lowering headroom; select a free GPU."
        )
    return {
        "free_gib": free_bytes / 2**30,
        "total_gib": total_bytes / 2**30,
        "required_gib": required_bytes / 2**30,
        "utilization": utilization,
        "headroom_gib": headroom_gib,
    }


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _load_resolved_config(main_repo: Path, overlay_path: Path, output_dir: Path):
    from b200_experiment.config import deep_merge, load_config, resolve_runtime_paths

    config = deep_merge(
        load_config(main_repo / "configs/qwen3_b200_cmt.yaml"),
        load_config(overlay_path),
    )
    config["experiment"]["output_dir"] = str(output_dir)
    config["experiment"]["seed"] = _env_int(
        "SEED", int(config["experiment"].get("seed", 42))
    )
    config["paths"]["storage_root"] = os.environ.get(
        "STORAGE_ROOT", config["paths"]["storage_root"]
    )
    if os.environ.get("STUDENT_MODEL"):
        config["models"]["student_path"] = os.environ["STUDENT_MODEL"]
    if os.environ.get("TEACHER_MODEL"):
        config["models"]["teacher_path"] = os.environ["TEACHER_MODEL"]
    if os.environ.get("TRAIN_DATA"):
        config["data"]["path"] = os.environ["TRAIN_DATA"]
    if os.environ.get("PROMPT_KEY"):
        config["data"]["prompt_key"] = os.environ["PROMPT_KEY"]
    if "TRAIN_DATA_SPLIT" in os.environ:
        value = os.environ["TRAIN_DATA_SPLIT"].strip()
        config["data"]["split"] = (
            None if value.lower() in {"", "none", "null"} else value
        )

    intervention = config["intervention"]
    intervention["num_probe_rollouts"] = _env_int(
        "NUM_PROBE_ROLLOUTS", intervention["num_probe_rollouts"]
    )
    intervention["future_horizon"] = _env_int(
        "FUTURE_HORIZON", intervention["future_horizon"]
    )
    intervention["min_original_suffix_tokens"] = _env_int(
        "MIN_ORIGINAL_SUFFIX_TOKENS", intervention["min_original_suffix_tokens"]
    )
    intervention["max_selection_attempts"] = _env_int(
        "MAX_SELECTION_ATTEMPTS", intervention["max_selection_attempts"]
    )
    intervention["matched_g_bins"] = _env_int(
        "MATCHED_G_BINS", intervention["matched_g_bins"]
    )
    config["training"]["max_steps"] = _env_int(
        "MAX_STEPS", int(config["training"]["max_steps"])
    )
    config["training"]["learning_rate"] = _env_float(
        "LEARNING_RATE", float(config["training"]["learning_rate"])
    )
    # These are protocol invariants, not user-tunable batching semantics.
    config["training"]["ppo_mini_batch_size"] = 1
    config["training"]["micro_batch_size_per_gpu"] = 1
    config["training"]["epochs"] = 1
    config["training"]["save_checkpoints"] = False
    config["training"]["save_optimizer"] = False
    config["training_evaluation"]["enabled"] = False
    config["selector"]["top_k"] = _env_int("TOP_K", int(config["selector"]["top_k"]))
    config["rollout"]["max_new_tokens"] = _env_int(
        "NORMAL_ROLLOUT_HORIZON", int(config["rollout"]["max_new_tokens"])
    )
    config["rollout"]["temperature"] = _env_float(
        "ROLLOUT_TEMPERATURE", float(config["rollout"]["temperature"])
    )
    config["rollout"]["top_p"] = _env_float(
        "ROLLOUT_TOP_P", float(config["rollout"]["top_p"])
    )
    vllm = config["rollout"]["vllm"]
    vllm["gpu_memory_utilization"] = _env_float(
        "ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION", vllm["gpu_memory_utilization"]
    )
    vllm["max_model_len"] = _env_int(
        "ROLLOUT_VLLM_MAX_MODEL_LEN", vllm["max_model_len"]
    )
    vllm["max_num_seqs"] = max(
        int(vllm.get("max_num_seqs", 4)), int(intervention["num_probe_rollouts"])
    )
    vllm["max_concurrent_requests"] = max(
        int(vllm.get("max_concurrent_requests", 4)),
        int(intervention["num_probe_rollouts"]),
    )
    return resolve_runtime_paths(config)


def _validate_config(config: dict[str, Any]) -> None:
    positive = {
        "MAX_STEPS": config["training"]["max_steps"],
        "NUM_PROBE_ROLLOUTS": config["intervention"]["num_probe_rollouts"],
        "FUTURE_HORIZON": config["intervention"]["future_horizon"],
        "TOP_K": config["selector"]["top_k"],
        "MIN_ORIGINAL_SUFFIX_TOKENS": config["intervention"][
            "min_original_suffix_tokens"
        ],
    }
    invalid = {key: value for key, value in positive.items() if int(value) <= 0}
    if invalid:
        raise ValueError(f"Experiment settings must be positive: {invalid}")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("LEARNING_RATE must be positive")
    if config["rollout"]["backend"] != "vllm":
        raise ValueError(
            "This intervention experiment requires the existing vLLM backend"
        )


def _score_cmt_rollout(student, teacher, rollout, config):
    from b200_experiment.scoring import score_student_teacher_rollout
    from b200_experiment.selectors import CMTSelector, PGTSelector

    selector = config["selector"]
    student_scores, teacher_scores = score_student_teacher_rollout(
        student,
        teacher,
        rollout,
        score_chunk_steps=int(selector.get("score_chunk_steps", 128)),
        top_k=int(selector["top_k"]),
        student_temperature=float(config["rollout"].get("temperature", 1.0)),
        teacher_temperature=float(
            config.get("opd", {}).get("teacher_temperature", 1.0)
        ),
        micro_batch_size=int(selector.get("score_micro_batch_size", 1)),
        trim_padding=bool(selector.get("trim_padding", True)),
        length_bucketed=bool(selector.get("length_bucketed_scoring", True)),
        compute_full_vocab_metrics=False,
    )
    if any(
        value is None
        for value in (
            student_scores.top_k_ids,
            student_scores.top_k_log_probs,
            student_scores.candidate_log_probs,
            teacher_scores.top_k_ids,
            teacher_scores.top_k_log_probs,
            teacher_scores.candidate_log_probs,
        )
    ):
        raise AssertionError("Joint scoring did not return every compact Top-K tensor")
    pgt = PGTSelector().compute_scores_from_topk(
        student_scores.top_k_ids,
        teacher_scores.top_k_ids,
        student_scores.top_k_log_probs,
        teacher_scores.candidate_log_probs,
        teacher_scores.top_k_log_probs,
        student_scores.candidate_log_probs,
        rollout.valid_mask,
        token_chunk_size=int(selector.get("pgt_vocab_chunk_tokens", 2048)),
        gain_support="student_topk",
    )
    cmt = CMTSelector(
        gamma=float(selector.get("cmt_gamma", 1.0)),
        successor_lambda=float(selector.get("cmt_successor_lambda", 1.0)),
        ablation_arm="g_d",
    ).compute_scores(pgt, rollout.response_ids, rollout.valid_mask)
    return student_scores, teacher_scores, pgt, cmt


def _slice_local_rollout(rollout, position_t: int):
    from b200_experiment.scoring import RolloutBatch

    stop = int(position_t) + 1
    input_stop = rollout.prompt_width + stop
    return RolloutBatch(
        input_ids=rollout.input_ids[:, :input_stop].clone(),
        attention_mask=rollout.attention_mask[:, :input_stop].clone(),
        response_ids=rollout.response_ids[:, :stop].clone(),
        valid_mask=rollout.valid_mask[:, :stop].clone(),
        rollout_log_probs=rollout.rollout_log_probs[:, :stop].clone(),
        prompt_width=rollout.prompt_width,
    )


def _local_fixed_support_kl_after(
    student, local_rollout, candidate_ids, teacher_logp, config
):
    from b200_experiment.scoring import score_original_rollout

    scores = score_original_rollout(
        student,
        local_rollout,
        keep_cache=False,
        score_chunk_steps=int(config["selector"].get("score_chunk_steps", 128)),
        retain_response_logits=False,
        top_k=0,
        candidate_ids=candidate_ids,
        temperature=float(config["rollout"].get("temperature", 1.0)),
        micro_batch_size=1,
        trim_padding=True,
        length_bucketed=False,
    )
    if scores.candidate_log_probs is None:
        raise AssertionError("Fixed-support post-update scoring returned no candidates")
    return fixed_support_reverse_kl(scores.candidate_log_probs[:, -1], teacher_logp)


def _probe_prefix_batch(prefix: torch.Tensor, count: int):
    ids = prefix.unsqueeze(0).repeat(int(count), 1)
    return ids, torch.ones_like(ids, dtype=torch.long)


def _run_probe_phase(
    phase: str,
    engine,
    student,
    teacher,
    prefix: torch.Tensor,
    paired_seeds: tuple[int, ...],
    event_id: int,
    config: dict[str, Any],
    audit_handle,
):
    count = len(paired_seeds)
    if tuple(paired_seeds) != paired_rollout_seeds(int(paired_seeds[0]), count):
        raise ValueError(
            "The existing vLLM batch API requires consecutive per-row probe seeds"
        )
    prompt_ids, prompt_mask = _probe_prefix_batch(prefix, count)
    max_model_len = int(config["rollout"]["vllm"]["max_model_len"])
    future_horizon = int(config["intervention"]["future_horizon"])
    if prefix.numel() + future_horizon > max_model_len:
        raise ValueError(
            f"Exact prefix ({prefix.numel()}) + FUTURE_HORIZON ({future_horizon}) "
            f"exceeds vLLM max_model_len={max_model_len}"
        )
    rollout = engine.generate(
        student,
        prompt_ids,
        prompt_mask,
        max_new_tokens=future_horizon,
        temperature=float(config["rollout"]["temperature"]),
        top_p=float(config["rollout"]["top_p"]),
        eos_token_ids=config["_tokenizer_eos_token_id"],
        pad_token_id=int(config["_tokenizer_pad_token_id"]),
        seed=int(paired_seeds[0]),
        sample_seed_offset=0,
    )
    student_scores, teacher_scores, pgt, _ = _score_cmt_rollout(
        student, teacher, rollout, config
    )
    kl = pgt.diagnostics["restricted_reverse_kl"].detach().float()
    student_topk_logp = student_scores.top_k_log_probs.detach().float()
    teacher_on_student_logp = teacher_scores.candidate_log_probs.detach().float()
    student_topk_weights = torch.softmax(student_topk_logp, dim=-1)
    # Existing only_stu/student_p OPD divergence proxy: the negative sum of
    # candidate rewards. Unlike restricted_reverse_kl, q is not separately
    # renormalized on Student Top-K, so these quantities are deliberately logged
    # as distinct measurements.
    opd_proxy = (
        student_topk_weights * (student_topk_logp - teacher_on_student_logp)
    ).sum(dim=-1)
    # Offset 0 scores s_t itself (the distribution that emits the first probe
    # token). Downstream state quality starts at s_{t+1}, after one newly
    # sampled action has actually changed the visited prefix.
    future_mask = rollout.valid_mask.clone()
    future_mask[:, 0] = False
    values = [float(value) for value in kl[future_mask].cpu().tolist()]
    opd_values = [float(value) for value in opd_proxy[future_mask].cpu().tolist()]
    if not values:
        raise RuntimeError(f"{phase} probe produced no valid future states")
    lengths = [int(value) for value in rollout.valid_mask.sum(dim=-1).cpu().tolist()]
    for row_index, seed in enumerate(paired_seeds):
        for offset in range(1, lengths[row_index]):
            record = {
                "event_id": int(event_id),
                "phase": phase,
                "rollout_seed": int(seed),
                "future_offset": int(offset),
                "restricted_reverse_kl": float(kl[row_index, offset].item()),
                "generated_token_id": int(
                    rollout.response_ids[row_index, offset].item()
                ),
            }
            assert_finite_values(record)
            audit_handle.write(json.dumps(record, allow_nan=False) + "\n")
    audit_handle.flush()
    return {
        "mean": fmean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
        "lengths": lengths,
        "opd_proxy_mean": fmean(opd_values),
    }


def _local_update(
    student,
    optimizer,
    rollout,
    student_scores,
    teacher_scores,
    position_t: int,
    config,
    device,
    distributed,
    event_id: int,
):
    from b200_experiment.opd_core import build_student_topk_opd_reference
    from b200_experiment.trainer import _opd_train_step

    local_rollout = _slice_local_rollout(rollout, position_t)
    local_t = local_rollout.response_ids.shape[1] - 1
    objective_mask = single_state_objective_mask(local_rollout.valid_mask, local_t)
    if int(objective_mask.sum().item()) != 1:
        raise AssertionError("Exactly one local state must enter the OPD objective")
    reference = build_student_topk_opd_reference(
        student_scores.top_k_ids[:, : local_t + 1],
        student_scores.top_k_log_probs[:, : local_t + 1],
        teacher_scores.candidate_log_probs[:, : local_t + 1],
        objective_mask,
        top_k=int(config["selector"]["top_k"]),
    )
    outside = ~objective_mask.unsqueeze(-1).expand_as(reference.advantages)
    if bool(reference.advantages[outside].ne(0).any()):
        raise AssertionError("Non-intervention states retained OPD advantages")
    weights = objective_mask.float()
    metrics = _opd_train_step(
        student,
        optimizer,
        local_rollout,
        weights,
        reference,
        config,
        device,
        distributed,
        objective_valid_mask=objective_mask,
        trajectory_active_mask=torch.ones(1, dtype=torch.bool, device=device),
        gibbs_scores=None,
        gibbs_epsilon=None,
        rollout_id=event_id,
        max_optimizer_steps=1,
        optimizer_step_start=event_id,
    )
    if int(metrics["optimizer_steps"]) != 1 or int(metrics["gibbs_allocations"]) != 0:
        raise AssertionError(
            "Local intervention must be one update with no Gibbs allocation"
        )
    return local_rollout, reference, metrics


def _append_jsonl(handle, payload: dict[str, Any]) -> None:
    assert_finite_values(payload)
    handle.write(json.dumps(payload, allow_nan=False) + "\n")
    handle.flush()


def run(args: argparse.Namespace) -> Path:
    main_repo = args.main_repo.resolve()
    if not (main_repo / "b200_experiment/trainer.py").is_file():
        raise FileNotFoundError(f"Invalid MAIN_REPO: {main_repo}")
    sys.path.insert(0, str(main_repo))

    from b200_experiment.config import save_config
    from b200_experiment.data import (
        epoch_batch_indices,
        filter_overlong_prompt_records,
        read_records,
        stable_sample_id,
        tokenize_prompts,
        validate_prompt_records,
    )
    from b200_experiment.distributed import DistributedContext
    from b200_experiment.models import load_models, load_student_tokenizer
    from b200_experiment.trainer import _make_optimizer, seed_everything
    from b200_experiment.vllm_rollout import VLLMRolloutEngine

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_resolved_config(main_repo, args.config.resolve(), output_dir)
    _validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("The B200 intervention experiment requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Experiment requires exactly one visible GPU per process; got "
            f"{torch.cuda.device_count()}. Set CUDA_VISIBLE_DEVICES to one device."
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    physical_gpu = _cuda_device_identity(0)
    distributed = DistributedContext(0, 0, 1, device)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    records, data_files = read_records(
        config["data"]["path"], split=config["data"].get("split")
    )
    validate_prompt_records(records, config["data"])
    prompt_tokenizer = load_student_tokenizer(config)
    records, filter_summary = filter_overlong_prompt_records(
        records, prompt_tokenizer, config["data"]
    )
    del prompt_tokenizer
    save_config(config, output_dir / "resolved_config.yaml")

    gpu_lease = exclusive_gpu_lease(physical_gpu)
    gpu_lease.__enter__()
    engine = VLLMRolloutEngine(config, output_dir)
    student = teacher = tokenizer = None
    try:
        memory_preflight = _validate_vllm_startup_memory(config)
        print(
            "GPU isolation: "
            f"physical={physical_gpu} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"free={memory_preflight['free_gib']:.1f}/"
            f"{memory_preflight['total_gib']:.1f} GiB "
            f"required={memory_preflight['required_gib']:.1f} GiB",
            flush=True,
        )
        engine.start()
        student, teacher, tokenizer, model_metadata = load_models(config, device)
        config["_tokenizer_eos_token_id"] = tokenizer.eos_token_id
        config["_tokenizer_pad_token_id"] = tokenizer.pad_token_id
        optimizer, fused = _make_optimizer(
            [
                parameter
                for parameter in student.parameters()
                if parameter.requires_grad
            ],
            config["training"],
        )
        metadata = {
            "protocol": "paired_exact_prefix_single_state_intervention_v1",
            "question": (
                "After training on a state s_t, does the distribution/quality of "
                "subsequent states visited by the student change, even when "
                "immediate gains are matched?"
            ),
            "main_repo": str(main_repo),
            "main_repo_commit": _git_commit(main_repo),
            "command": sys.argv,
            "model": model_metadata,
            "data_files": [str(path) for path in data_files],
            "data_filter": filter_summary,
            "optimizer_fused": bool(fused),
            "selection": "uniform_over_eligible_positions_without_D_or_future_outcome",
            "state_definition": "s_t=prompt+y_<t; y_t excluded",
            "local_update": "one_student_topk_OPD_state_weight_1_no_Gibbs",
            "future_metric": "conditional_reverse_KL_on_each_visited_state_student_topk",
            "paired_seeds": True,
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

        events_path = output_dir / "events.jsonl"
        audit_path = output_dir / "future_states.jsonl.gz"
        max_steps = int(config["training"]["max_steps"])
        intervention = config["intervention"]
        num_probes = int(intervention["num_probe_rollouts"])
        min_suffix = int(intervention["min_original_suffix_tokens"])
        max_attempts = int(intervention["max_selection_attempts"])
        with contextlib.ExitStack() as output_stack:
            events_handle = output_stack.enter_context(
                events_path.open("w", encoding="utf-8")
            )
            audit_handle = output_stack.enter_context(
                gzip.open(audit_path, "wt", encoding="utf-8")
            )
            for event_id in range(max_steps):
                selected = None
                for attempt in range(max_attempts):
                    schedule_step = event_id * max_attempts + attempt
                    record_index = epoch_batch_indices(
                        len(records), 1, schedule_step, seed
                    )[0]
                    record = records[record_index]
                    encoded, _ = tokenize_prompts(
                        [record], tokenizer, config["data"], device
                    )
                    normal_seed = seed + 1_000_000 + schedule_step
                    rollout = engine.generate(
                        student,
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        max_new_tokens=int(config["rollout"]["max_new_tokens"]),
                        temperature=float(config["rollout"]["temperature"]),
                        top_p=float(config["rollout"]["top_p"]),
                        eos_token_ids=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        seed=normal_seed,
                    )
                    response_length = int(rollout.valid_mask[0].sum().item())
                    try:
                        position_t = select_state_position(
                            response_length,
                            min_suffix,
                            seed=seed + 2_000_000 + schedule_step,
                        )
                    except ValueError:
                        continue
                    selected = (
                        record_index,
                        record,
                        encoded,
                        rollout,
                        response_length,
                        position_t,
                        normal_seed,
                    )
                    break
                if selected is None:
                    raise RuntimeError(
                        f"Could not find an eligible trajectory for event {event_id} "
                        f"after {max_attempts} on-policy attempts"
                    )
                (
                    record_index,
                    record,
                    encoded,
                    rollout,
                    response_length,
                    position_t,
                    normal_seed,
                ) = selected
                student_scores, teacher_scores, pgt, cmt = _score_cmt_rollout(
                    student, teacher, rollout, config
                )
                diagnostics = cmt.diagnostics
                prefix = build_state_prefix(
                    encoded["input_ids"][0],
                    encoded["attention_mask"][0],
                    rollout.response_ids[0, :response_length],
                    position_t,
                )
                probe_seed_base = seed + 10_000_000 + event_id * 10_000
                probe_seeds = paired_rollout_seeds(probe_seed_base, num_probes)
                prefix_before = prefix.detach().clone()
                before = _run_probe_phase(
                    "before",
                    engine,
                    student,
                    teacher,
                    prefix_before,
                    probe_seeds,
                    event_id,
                    config,
                    audit_handle,
                )

                local_kl_before = float(
                    pgt.diagnostics["restricted_reverse_kl"][0, position_t].item()
                )
                local_rollout, reference, update = _local_update(
                    student,
                    optimizer,
                    rollout,
                    student_scores,
                    teacher_scores,
                    position_t,
                    config,
                    device,
                    distributed,
                    event_id,
                )
                fixed_pre = fixed_support_reverse_kl(
                    reference.old_student_log_probs[:, -1],
                    reference.teacher_log_probs[:, -1],
                )
                if not torch.allclose(
                    fixed_pre,
                    torch.tensor([local_kl_before], device=fixed_pre.device),
                    atol=2e-5,
                    rtol=2e-5,
                ):
                    raise AssertionError(
                        "Pre-update fixed-support KL disagrees with CMT restricted_reverse_kl"
                    )
                local_kl_after_tensor = _local_fixed_support_kl_after(
                    student,
                    local_rollout,
                    reference.candidate_ids,
                    reference.teacher_log_probs[:, -1],
                    config,
                )
                local_kl_after = float(local_kl_after_tensor.item())

                prefix_after = prefix.detach().clone()
                assert_paired_probe_protocol(
                    prefix_before, prefix_after, probe_seeds, probe_seeds
                )
                after = _run_probe_phase(
                    "after",
                    engine,
                    student,
                    teacher,
                    prefix_after,
                    probe_seeds,
                    event_id,
                    config,
                    audit_handle,
                )
                event = {
                    "event_id": event_id,
                    "step": event_id + 1,
                    "dataset_id": Path(config["data"]["path"]).name,
                    "dataset_index": record_index,
                    "sample_id": stable_sample_id(record, record_index),
                    "normal_rollout_seed": normal_seed,
                    "response_position_t": position_t,
                    "predicted_action_y_t": int(
                        rollout.response_ids[0, position_t].item()
                    ),
                    "prefix_length": int(prefix.numel()),
                    "state_prefix_token_ids": [
                        int(token) for token in prefix.detach().cpu().tolist()
                    ],
                    "local_fixed_support_token_ids": [
                        int(token)
                        for token in reference.candidate_ids[0, -1]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "local_student_log_probs_pre": [
                        float(value)
                        for value in reference.old_student_log_probs[0, -1]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "local_teacher_log_probs_fixed": [
                        float(value)
                        for value in reference.teacher_log_probs[0, -1]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "original_trajectory_length": response_length,
                    "original_suffix_tokens_after_y_t": (
                        response_length - position_t - 1
                    ),
                    "g_t": float(diagnostics["gain"][0, position_t].item()),
                    "D_t": float(
                        diagnostics["sequential_gain_raw"][0, position_t].item()
                    ),
                    "sequential_gain_raw": float(
                        diagnostics["sequential_gain_raw"][0, position_t].item()
                    ),
                    "successor_excess": float(
                        diagnostics["successor_excess"][0, position_t].item()
                    ),
                    "marginal_flux": float(
                        diagnostics["marginal_flux"][0, position_t].item()
                    ),
                    "learning_value_raw": float(
                        diagnostics["learning_value_raw"][0, position_t].item()
                    ),
                    "local_kl_before": local_kl_before,
                    "local_kl_after": local_kl_after,
                    "realized_local_improvement": local_kl_before - local_kl_after,
                    "future_kl_before_mean": before["mean"],
                    "future_kl_before_std": before["std"],
                    "future_kl_after_mean": after["mean"],
                    "future_kl_after_std": after["std"],
                    "future_improvement": before["mean"] - after["mean"],
                    "future_opd_proxy_before_mean": before["opd_proxy_mean"],
                    "future_opd_proxy_after_mean": after["opd_proxy_mean"],
                    "number_future_valid_states": {
                        "before": before["count"],
                        "after": after["count"],
                    },
                    "number_future_valid_states_before": before["count"],
                    "number_future_valid_states_after": after["count"],
                    "rollout_lengths": {
                        "before": before["lengths"],
                        "after": after["lengths"],
                    },
                    "paired_rollout_seeds": list(probe_seeds),
                    "learning_rate": float(config["training"]["learning_rate"]),
                    "gradient_norm": float(update["gradient_norm"]),
                    "local_update_loss": float(update["loss"]),
                    "local_update_base_topk_opd_loss": float(
                        update["base_topk_opd_loss"]
                    ),
                    "local_update_supervised_states": 1,
                    "local_update_optimizer_steps": int(update["optimizer_steps"]),
                    "local_update_gibbs_allocations": int(update["gibbs_allocations"]),
                    "top_k": int(config["selector"]["top_k"]),
                    "selection_rule": "uniform_eligible_without_D_or_future",
                    "prefix_sha256": hashlib.sha256(
                        prefix.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
                _append_jsonl(events_handle, event)
                print(
                    f"event={event_id + 1}/{max_steps} sample={event['sample_id']} "
                    f"t={position_t} g={event['g_t']:.6g} "
                    f"D={event['D_t']:.6g} local_delta="
                    f"{event['realized_local_improvement']:.6g} future_delta="
                    f"{event['future_improvement']:.6g}",
                    flush=True,
                )
        analyze_output(output_dir, int(intervention["matched_g_bins"]))
    finally:
        try:
            engine.close()
        finally:
            try:
                distributed.close()
            finally:
                gpu_lease.__exit__(None, None, None)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-repo", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = run(args)
    print(f"Completed intervention experiment: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
