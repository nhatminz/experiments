#!/usr/bin/env python3
"""Run independent single-state interventions from one fixed base checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import yaml

from .gpu_isolation import exclusive_gpu_lease
from .plotting import analyze_output
from .accuracy_analysis import analyze_accuracy, intervention_accuracy_fields
from .protocol import (
    assert_same_prefix_independent_branches,
    branch_seed_sets,
    build_state_prefix,
    descendant_offsets,
    fixed_support_reverse_kl,
    future_distribution_stats,
    kl_scoring_token_count,
    reconstruct_assistant_response,
    resolve_reference_answer,
    select_state_position,
)
from .utils import (
    ProgressLogger,
    append_gzip_jsonl,
    append_jsonl,
    environment_payload,
    read_jsonl,
    pending_candidates,
    restore_parameters_exact,
    sanitize_continuation_audit,
    snapshot_parameters,
    validate_intervention_schema,
)


def _env_int(name: str, default: int, *aliases: str) -> int:
    for key in (name, *aliases):
        if key in os.environ:
            return int(os.environ[key])
    return int(default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    if name not in os.environ:
        return bool(default)
    value = os.environ[name].strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {os.environ[name]!r}")


def _eos_ids(tokenizer) -> list[int]:
    value = tokenizer.eos_token_id
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _load_resolved_config(main_repo: Path, overlay: Path, output_dir: Path):
    from b200_experiment.config import deep_merge, load_config, resolve_runtime_paths

    config = deep_merge(
        load_config(main_repo / "configs/qwen3_b200_base.yaml"),
        load_config(overlay),
    )
    config["experiment"]["output_dir"] = str(output_dir)
    config["paths"]["storage_root"] = os.environ.get(
        "STORAGE_ROOT", config["paths"]["storage_root"]
    )
    for env_name, section, key in (
        ("STUDENT_MODEL", "models", "student_path"),
        ("TEACHER_MODEL", "models", "teacher_path"),
        ("TRAIN_DATA", "data", "path"),
        ("PROMPT_KEY", "data", "prompt_key"),
    ):
        if os.environ.get(env_name):
            config[section][key] = os.environ[env_name]
    if "TRAIN_DATA_SPLIT" in os.environ:
        value = os.environ["TRAIN_DATA_SPLIT"].strip()
        config["data"]["split"] = None if value.lower() in {"", "none", "null"} else value

    config["experiment"]["seed"] = _env_int(
        "SEED", config["experiment"].get("seed", 42)
    )
    settings = config["intervention"]
    settings["num_states"] = _env_int(
        "NUM_STATES", settings["num_states"], "MAX_STEPS"
    )
    settings["num_continuations"] = _env_int(
        "NUM_CONTINUATIONS", settings["num_continuations"], "K_ROLLOUTS",
        "NUM_PROBE_ROLLOUTS"
    )
    settings["future_horizon"] = _env_int(
        "FUTURE_HORIZON", settings["future_horizon"]
    )
    settings["top_k"] = _env_int("TOP_K", settings["top_k"])
    settings["min_original_suffix_tokens"] = _env_int(
        "MIN_ORIGINAL_SUFFIX_TOKENS", settings["min_original_suffix_tokens"]
    )
    settings["min_response_position"] = _env_int(
        "MIN_RESPONSE_POSITION", settings.get("min_response_position", 8)
    )
    settings["candidate_collection_batch_size"] = _env_int(
        "CANDIDATE_COLLECTION_BATCH_SIZE", settings["candidate_collection_batch_size"]
    )
    for env_name, key in (
        ("MATCHED_QUANTILE_LOW", "matched_quantile_low"),
        ("MATCHED_QUANTILE_HIGH", "matched_quantile_high"),
        ("SENSITIVITY_QUANTILE_LOW", "sensitivity_quantile_low"),
        ("SENSITIVITY_QUANTILE_HIGH", "sensitivity_quantile_high"),
    ):
        settings[key] = _env_float(env_name, settings[key])
    config["training"]["learning_rate"] = _env_float(
        "LEARNING_RATE", config["training"]["learning_rate"]
    )
    config["rollout"]["max_new_tokens"] = _env_int(
        "CANDIDATE_ROLLOUT_HORIZON", config["rollout"]["max_new_tokens"],
        "NORMAL_ROLLOUT_HORIZON"
    )
    config["rollout"]["temperature"] = _env_float(
        "ROLLOUT_TEMPERATURE", config["rollout"]["temperature"]
    )
    config["rollout"]["top_p"] = _env_float(
        "ROLLOUT_TOP_P", config["rollout"]["top_p"]
    )
    vllm = config["rollout"]["vllm"]
    vllm["gpu_memory_utilization"] = _env_float(
        "ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION", vllm["gpu_memory_utilization"]
    )
    vllm["max_model_len"] = _env_int(
        "ROLLOUT_VLLM_MAX_MODEL_LEN", vllm["max_model_len"]
    )
    needed = max(
        settings["num_continuations"], settings["candidate_collection_batch_size"]
    )
    vllm["max_num_seqs"] = max(int(vllm.get("max_num_seqs", 1)), needed)
    vllm["max_concurrent_requests"] = max(
        int(vllm.get("max_concurrent_requests", 1)), needed
    )
    accuracy = config["accuracy_analysis"]
    accuracy["enabled"] = _env_bool(
        "ENABLE_ACCURACY_ANALYSIS", accuracy.get("enabled", True)
    )
    accuracy["max_new_tokens"] = _env_int(
        "ACCURACY_MAX_NEW_TOKENS", accuracy["max_new_tokens"]
    )
    accuracy["bootstrap_replicates"] = _env_int(
        "BOOTSTRAP_REPLICATES", accuracy["bootstrap_replicates"]
    )
    config["experiment"]["accuracy_analysis_enabled"] = accuracy["enabled"]
    return resolve_runtime_paths(config)


def _validate_config(config: dict[str, Any]) -> None:
    settings = config["intervention"]
    positive = {
        "num_states": settings["num_states"],
        "num_continuations": settings["num_continuations"],
        "future_horizon": settings["future_horizon"],
        "candidate_collection_batch_size": settings["candidate_collection_batch_size"],
    }
    if any(int(value) <= 0 for value in positive.values()):
        raise ValueError(f"Experiment settings must be positive: {positive}")
    if int(settings["top_k"]) != 16:
        raise ValueError("The canonical experiment fixes local support to Student Top-16")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if config["rollout"]["backend"] != "vllm":
        raise ValueError("This experiment requires the existing vLLM rollout backend")
    for low_key, high_key in (
        ("matched_quantile_low", "matched_quantile_high"),
        ("sensitivity_quantile_low", "sensitivity_quantile_high"),
    ):
        low, high = float(settings[low_key]), float(settings[high_key])
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"Invalid quantile interval: {low_key}={low}, {high_key}={high}")
    accuracy = config["accuracy_analysis"]
    if int(accuracy["max_new_tokens"]) <= 0:
        raise ValueError("accuracy_analysis.max_new_tokens must be positive")
    if int(accuracy["bootstrap_replicates"]) <= 0:
        raise ValueError("accuracy_analysis.bootstrap_replicates must be positive")


def _resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that would alter an already-started scientific run."""
    return {
        "protocol": config["experiment"]["protocol"],
        "experiment_schema_version": config["experiment"]["experiment_schema_version"],
        "accuracy_analysis_enabled": config["experiment"]["accuracy_analysis_enabled"],
        "seed": config["experiment"]["seed"],
        "models": config["models"],
        "data": config["data"],
        "sampling": {
            key: config["rollout"][key]
            for key in ("max_new_tokens", "temperature", "top_p")
        },
        "intervention": config["intervention"],
        "accuracy_analysis": config["accuracy_analysis"],
        "local_optimizer": {
            key: config["training"][key]
            for key in ("learning_rate", "adam_betas", "weight_decay", "max_grad_norm")
        },
    }


def _assert_resume_schema(previous: dict[str, Any], current: dict[str, Any]) -> None:
    previous_version = previous.get("experiment", {}).get(
        "experiment_schema_version", 1
    )
    current_version = current["experiment"]["experiment_schema_version"]
    if int(previous_version) != int(current_version):
        raise ValueError(
            "Incompatible Experiment output schema: this accuracy-enabled "
            "runner requires schema version 2. Use a new RUN_NAME/output directory."
        )


def _validate_completed_accuracy_records(
    rows: list[dict[str, Any]], *, enabled: bool
) -> None:
    if not enabled:
        return
    required = {
        "accuracy_before",
        "accuracy_after",
        "delta_accuracy",
        "pass8_before",
        "pass8_after",
        "delta_pass8",
        "num_correct_before",
        "num_correct_after",
    }
    incompatible = [
        int(row.get("intervention_id", -1))
        for row in rows
        if not required.issubset(row)
        or int(row.get("experiment_schema_version", 1)) != 2
    ]
    if incompatible:
        raise ValueError(
            "Completed records lack schema-v2 full-generation accuracy fields "
            f"(first IDs: {incompatible[:5]}). Use a new RUN_NAME."
        )


def _cuda_identity() -> str:
    properties = torch.cuda.get_device_properties(0)
    value = getattr(properties, "uuid", None) or getattr(properties, "pci_bus_id", None)
    return str(value or f"{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}-{properties.name}")


def _prefix_batch(prefix: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = prefix.unsqueeze(0).repeat(int(count), 1)
    return ids, torch.ones_like(ids, dtype=torch.long)


def _generate_suffixes(engine, model, prefix, seeds, config, tokenizer):
    if tuple(seeds) != tuple(range(int(seeds[0]), int(seeds[0]) + len(seeds))):
        raise ValueError("vLLM batch generation requires consecutive row seeds")
    prompt_ids, prompt_mask = _prefix_batch(prefix, len(seeds))
    horizon = int(config["intervention"]["future_horizon"])
    # Generate once for both measurements. KL later consumes only horizon+1
    # tokens (offset zero plus at most `horizon` descendants); grading sees all.
    generation_tokens = (
        int(config["accuracy_analysis"]["max_new_tokens"])
        if bool(config["accuracy_analysis"]["enabled"])
        else horizon + 1
    )
    if prefix.numel() + generation_tokens > int(config["rollout"]["vllm"]["max_model_len"]):
        raise ValueError(
            "State prefix plus accuracy generation horizon exceeds vLLM max_model_len"
        )
    with torch.inference_mode():
        return engine.generate(
            model,
            prompt_ids,
            prompt_mask,
            max_new_tokens=generation_tokens,
            temperature=float(config["rollout"]["temperature"]),
            top_p=float(config["rollout"]["top_p"]),
            eos_token_ids=_eos_ids(tokenizer),
            pad_token_id=int(tokenizer.pad_token_id),
            seed=int(seeds[0]),
        )


def _next_token_logits(model, prefix: torch.Tensor) -> torch.Tensor:
    attention = torch.ones_like(prefix, dtype=torch.long).unsqueeze(0)
    output = model(input_ids=prefix.unsqueeze(0), attention_mask=attention, use_cache=False)
    return output.logits[:, -1, :]


def _single_local_update(student, teacher, prefix, config):
    """One AdamW step on D0 itself, on one fixed Student-Top-16 support."""
    top_k = int(config["intervention"]["top_k"])
    student.eval()
    teacher.eval()
    with torch.inference_mode():
        base_student_logits = _next_token_logits(student, prefix)
        support_ids = torch.topk(base_student_logits, k=top_k, dim=-1).indices
        teacher_logits = _next_token_logits(teacher, prefix)
        fixed_teacher = teacher_logits.gather(-1, support_ids).detach()
        fixed_student_before = base_student_logits.gather(-1, support_ids).detach()
        p_before = torch.softmax(fixed_student_before.float(), dim=-1)
        q_fixed = torch.softmax(fixed_teacher.float(), dim=-1)
        d0_value = float(
            fixed_support_reverse_kl(fixed_student_before, fixed_teacher).item()
        )

    # Tensors created inside inference_mode cannot be saved by autograd. Make
    # ordinary immutable copies for the one differentiable local forward.
    support_ids = support_ids.clone()
    fixed_teacher = fixed_teacher.clone()

    # Keep evaluation mode so the differentiable D0 is the exact same
    # categorical object measured above (gradients remain enabled in eval mode).
    student.eval()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(float(x) for x in config["training"]["adam_betas"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=bool(torch.cuda.is_available()),
    )
    optimizer.zero_grad(set_to_none=True)
    logits = _next_token_logits(student, prefix).gather(-1, support_ids)
    loss = fixed_support_reverse_kl(logits, fixed_teacher).mean()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable, float(config["training"]["max_grad_norm"])
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    student.eval()
    with torch.inference_mode():
        fixed_student_after = _next_token_logits(student, prefix).gather(-1, support_ids)
        d1_value = float(
            fixed_support_reverse_kl(fixed_student_after, fixed_teacher).item()
        )
    return {
        "support_ids": support_ids[0].detach().cpu().tolist(),
        "p_base_conditional": p_before[0].detach().cpu().tolist(),
        "q_teacher_conditional": q_fixed[0].detach().cpu().tolist(),
        "d0": d0_value,
        "d1": d1_value,
        "delta_immediate": d0_value - d1_value,
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "optimizer": optimizer,
    }


def _slice_rollout_for_kl(rollout, future_horizon: int):
    from b200_experiment.scoring import RolloutBatch

    stop = min(
        rollout.response_ids.shape[1], int(future_horizon) + 1
    )
    return RolloutBatch(
        input_ids=rollout.input_ids[:, : rollout.prompt_width + stop],
        attention_mask=rollout.attention_mask[:, : rollout.prompt_width + stop],
        response_ids=rollout.response_ids[:, :stop],
        valid_mask=rollout.valid_mask[:, :stop],
        rollout_log_probs=rollout.rollout_log_probs[:, :stop],
        prompt_width=rollout.prompt_width,
    )


def _score_suffixes_with_base(
    student, teacher, rollout, seeds, phase, config, tokenizer, candidate
):
    from b200_experiment.evaluation import grade_evaluation_response
    from b200_experiment.scoring import score_student_teacher_rollout

    selector = config["selector"]
    future_horizon = int(config["intervention"]["future_horizon"])
    kl_rollout = _slice_rollout_for_kl(rollout, future_horizon)
    with torch.inference_mode():
        student_scores, teacher_scores = score_student_teacher_rollout(
            student,
            teacher,
            kl_rollout,
            score_chunk_steps=int(selector.get("score_chunk_steps", 128)),
            top_k=int(config["intervention"]["top_k"]),
            student_temperature=1.0,
            teacher_temperature=1.0,
            micro_batch_size=int(selector.get("score_micro_batch_size", 8)),
            trim_padding=bool(selector.get("trim_padding", True)),
            length_bucketed=bool(selector.get("length_bucketed_scoring", True)),
            compute_full_vocab_metrics=False,
        )
    if student_scores.top_k_log_probs is None or teacher_scores.candidate_log_probs is None:
        raise AssertionError("Compact Student-TopK scoring did not return candidate values")
    quality = fixed_support_reverse_kl(
        student_scores.top_k_log_probs, teacher_scores.candidate_log_probs
    ).detach().float()
    branches: list[list[float]] = []
    audit: list[dict[str, Any]] = []
    lengths = [int(x) for x in rollout.valid_mask.sum(dim=-1).cpu().tolist()]
    kl_lengths = [
        kl_scoring_token_count(length, future_horizon) for length in lengths
    ]
    eos_ids = set(_eos_ids(tokenizer))
    response_prefix_ids = [int(item) for item in candidate["response_prefix_token_ids"]]
    reference_answer = str(candidate["reference_answer"])
    for row, (seed, length, kl_length) in enumerate(zip(seeds, lengths, kl_lengths)):
        # Offset zero is s_t itself. Only states after >=1 generated action are used.
        offsets = descendant_offsets(kl_length)[:future_horizon]
        values = [float(quality[row, offset].item()) for offset in offsets]
        branches.append(values)
        generated_ids = [
            int(x) for x in rollout.response_ids[row, :length].detach().cpu().tolist()
        ]
        reconstructed = reconstruct_assistant_response(
            tokenizer, response_prefix_ids, generated_ids
        )
        correct = bool(grade_evaluation_response(
            reconstructed,
            {"answer": reference_answer},
            benchmark=str(config["accuracy_analysis"]["benchmark"]),
        ))
        generation_limit = (
            int(config["accuracy_analysis"]["max_new_tokens"])
            if bool(config["accuracy_analysis"]["enabled"])
            else future_horizon + 1
        )
        # The OpenAI-compatible vLLM endpoint can omit the matched stop token
        # from ``token_ids``.  EOS is the only configured stop condition here,
        # so an otherwise-short response is also an EOS termination.
        terminated_by_eos = bool(
            (generated_ids and generated_ids[-1] in eos_ids)
            or length < generation_limit
        )
        truncated = bool(
            not terminated_by_eos
            and length >= generation_limit
        )
        audit.append({
            "phase": phase,
            "branch": phase,
            "continuation_index": row,
            "rollout_index": row,
            "seed": int(seed),
            "rng_seed": int(seed),
            "generated_token_ids": [
                int(x) for x in rollout.response_ids[row, :length].detach().cpu().tolist()
            ],
            "generated_length": length,
            "decoded_continuation": tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
            ),
            "generated_continuation_text": tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ),
            "assistant_prefix_text": candidate["response_prefix_text"],
            "reconstructed_full_response": reconstructed,
            "reconstructed_response": reconstructed,
            "reference_answer": reference_answer,
            "predicted_correct": correct,
            "correct": correct,
            "terminated_by_eos": terminated_by_eos,
            "truncated_at_accuracy_limit": truncated,
            "accuracy_max_new_tokens": generation_limit,
            "future_kl_horizon": future_horizon,
            "descendant_quality": values,
            "descendant_offsets": offsets,
            "num_evaluated_descendant_states": len(values),
            "num_future_states_scored": len(values),
            "mean_future_reverse_kl": (
                sum(values) / len(values) if values else None
            ),
            "future_kl_mean": sum(values) / len(values) if values else None,
            "future_kl_std": (
                float(torch.tensor(values).std(unbiased=False).item())
                if values else None
            ),
        })
    return future_distribution_stats(branches), audit, lengths


def _collect_candidates(
    path: Path, records, tokenizer, student, engine, config, device, log
) -> list[dict[str, Any]]:
    from b200_experiment.data import stable_sample_id, tokenize_prompts

    target = int(config["intervention"]["num_states"])
    candidates = read_jsonl(path)
    if len(candidates) > target:
        raise ValueError("candidate_states.jsonl has more rows than configured num_states")
    if len(candidates) == target:
        log(f"Candidate collection already complete: {target}/{target}")
        return candidates
    used = {int(row["dataset_index"]) for row in candidates}
    seed = int(config["experiment"]["seed"])
    order = list(range(len(records)))
    random.Random(seed + 101).shuffle(order)
    batch_size = int(config["intervention"]["candidate_collection_batch_size"])
    eos_ids = _eos_ids(tokenizer)
    progress_path = path.with_name("candidate_collection_progress.json")
    cursor = 0
    if progress_path.is_file():
        cursor = int(json.loads(progress_path.read_text(encoding="utf-8"))["cursor"])
    while len(candidates) < target and cursor < len(order):
        batch_begin = cursor
        indices = order[cursor : cursor + batch_size]
        cursor += len(indices)
        batch_records = [records[index] for index in indices]
        encoded, _ = tokenize_prompts(batch_records, tokenizer, config["data"], device)
        generation_seed = seed + 1_000_000 + batch_begin * 100
        with torch.inference_mode():
            rollout = engine.generate(
                student,
                encoded["input_ids"],
                encoded["attention_mask"],
                max_new_tokens=int(config["rollout"]["max_new_tokens"]),
                temperature=float(config["rollout"]["temperature"]),
                top_p=float(config["rollout"]["top_p"]),
                eos_token_ids=eos_ids,
                pad_token_id=int(tokenizer.pad_token_id),
                seed=generation_seed,
            )
        for row, (dataset_index, record) in enumerate(zip(indices, batch_records)):
            if dataset_index in used:
                continue
            length = int(rollout.valid_mask[row].sum().item())
            tokens = rollout.response_ids[row, :length]
            token_list = [int(x) for x in tokens.detach().cpu().tolist()]
            try:
                position = select_state_position(
                    token_list,
                    eos_token_ids=eos_ids,
                    min_suffix_tokens=int(config["intervention"]["min_original_suffix_tokens"]),
                    min_position=int(config["intervention"].get("min_response_position", 8)),
                    seed=seed + 2_000_000 + dataset_index,
                )
            except ValueError:
                continue
            prefix = build_state_prefix(
                encoded["input_ids"][row], encoded["attention_mask"][row], tokens, position
            )
            generation_horizon = (
                int(config["accuracy_analysis"]["max_new_tokens"])
                if bool(config["accuracy_analysis"]["enabled"])
                else int(config["intervention"]["future_horizon"]) + 1
            )
            if prefix.numel() + generation_horizon > int(
                config["rollout"]["vllm"]["max_model_len"]
            ):
                continue
            prompt_tokens = encoded["input_ids"][row][
                encoded["attention_mask"][row].bool()
            ]
            response_prefix_ids = token_list[:position]
            if prefix[int(prompt_tokens.numel()):].detach().cpu().tolist() != response_prefix_ids:
                raise AssertionError("Assistant response prefix boundary is incorrect")
            reference_answer = resolve_reference_answer(
                record, config["accuracy_analysis"].get("answer_key", "answer")
            )
            payload = {
                "candidate_id": len(candidates),
                "dataset_index": int(dataset_index),
                "sample_id": stable_sample_id(record, dataset_index),
                "state_position_t": int(position),
                "prefix_token_ids": [int(x) for x in prefix.detach().cpu().tolist()],
                "decoded_prefix": tokenizer.decode(
                    prefix.detach().cpu().tolist(), skip_special_tokens=False
                ),
                "prefix_length": int(prefix.numel()),
                "response_start_token_index": int(prompt_tokens.numel()),
                "response_prefix_token_ids": response_prefix_ids,
                "response_prefix_text": tokenizer.decode(
                    response_prefix_ids, skip_special_tokens=True
                ),
                "problem_text": str(record.get(config["data"]["prompt_key"], "")),
                "reference_answer": reference_answer,
                "trajectory_length": length,
                "candidate_rollout_seed": generation_seed + row,
                "action_y_t": token_list[position],
                "selection": "uniform_valid_interior_position_without_quality_signal",
            }
            append_jsonl(path, payload)
            candidates.append(payload)
            used.add(dataset_index)
            if len(candidates) >= target:
                break
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"cursor": cursor, "candidate_count": len(candidates)}) + "\n",
            encoding="utf-8",
        )
        temporary.replace(progress_path)
        log(f"Collected candidate states: {len(candidates)}/{target}")
    if len(candidates) != target:
        raise RuntimeError(
            f"Only collected {len(candidates)}/{target} valid states from distinct prompts"
        )
    if len({row["dataset_index"] for row in candidates}) != len(candidates):
        raise AssertionError("Candidate states are not from distinct prompts")
    return candidates


def run(args: argparse.Namespace) -> Path:
    main_repo = args.main_repo.resolve()
    if not (main_repo / "b200_experiment/models.py").is_file():
        raise FileNotFoundError(f"Invalid MAIN_REPO: {main_repo}")
    if str(main_repo) not in sys.path:
        sys.path.insert(0, str(main_repo))
    from b200_experiment.config import save_config
    from b200_experiment.data import (
        filter_overlong_prompt_records,
        read_records,
        validate_prompt_records,
    )
    from b200_experiment.models import load_models, load_student_tokenizer
    from b200_experiment.trainer import seed_everything
    from b200_experiment.vllm_rollout import VLLMRolloutEngine
    from torch.utils.tensorboard import SummaryWriter

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_resolved_config(main_repo, args.config.resolve(), output_dir)
    if args.num_states is not None:
        config["intervention"]["num_states"] = int(args.num_states)
    if args.k_rollouts is not None:
        config["intervention"]["num_continuations"] = int(args.k_rollouts)
    if args.future_horizon is not None:
        config["intervention"]["future_horizon"] = int(args.future_horizon)
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = float(args.learning_rate)
    if args.accuracy_max_new_tokens is not None:
        config["accuracy_analysis"]["max_new_tokens"] = int(
            args.accuracy_max_new_tokens
        )
    if args.enable_accuracy_analysis is not None:
        config["accuracy_analysis"]["enabled"] = bool(
            args.enable_accuracy_analysis
        )
        config["experiment"]["accuracy_analysis_enabled"] = bool(
            args.enable_accuracy_analysis
        )
    if args.bootstrap_replicates is not None:
        config["accuracy_analysis"]["bootstrap_replicates"] = int(
            args.bootstrap_replicates
        )
    _validate_config(config)
    log = ProgressLogger(output_dir / "progress.log")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one B200 GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    records, data_files = read_records(config["data"]["path"], config["data"].get("split"))
    validate_prompt_records(records, config["data"])
    light_tokenizer = load_student_tokenizer(config)
    records, filter_summary = filter_overlong_prompt_records(records, light_tokenizer, config["data"])
    del light_tokenizer
    resolved_yaml = output_dir / "resolved_config.yaml"
    if resolved_yaml.exists():
        previous = yaml.safe_load(resolved_yaml.read_text(encoding="utf-8"))
        _assert_resume_schema(previous, config)
        if _resume_signature(previous) != _resume_signature(config):
            raise ValueError(
                "Refusing to resume with a changed scientific configuration; "
                "use the original settings or a new RUN_NAME"
            )
    else:
        save_config(config, resolved_yaml)
    resolved_json = output_dir / "resolved_config.json"
    if not resolved_json.exists():
        resolved_json.write_text(
            json.dumps({k: v for k, v in config.items() if not k.startswith("_")}, indent=2) + "\n",
            encoding="utf-8",
        )
    environment_path = output_dir / "environment.json"
    if not environment_path.exists():
        environment_path.write_text(
            json.dumps(environment_payload(main_repo), indent=2) + "\n", encoding="utf-8"
        )

    identity = _cuda_identity()
    writer = SummaryWriter(output_dir / config["logging"]["tensorboard"]["log_dir"])
    engine = VLLMRolloutEngine(config, output_dir)
    student = teacher = tokenizer = None
    with exclusive_gpu_lease(identity):
        try:
            log(f"Protocol: {config['experiment']['protocol']} (no CMT/LIFT quantities)")
            log(f"Models: teacher={config['models']['teacher_path']} student={config['models']['student_path']}")
            log(f"Data: {data_files}; kept={filter_summary['kept_count']}")
            engine.start()
            student, teacher, tokenizer, model_metadata = load_models(config, device)
            student.eval()
            teacher.eval()
            base_snapshot = snapshot_parameters(student)
            log(f"Captured one fixed BF16 base snapshot with {len(base_snapshot)} tensors")
            metadata_path = output_dir / "environment.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({"model": model_metadata, "data_filter": filter_summary})
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            candidates = _collect_candidates(
                output_dir / "candidate_states.jsonl", records, tokenizer, student,
                engine, config, device, log
            )
            # This line is reached only after every state was generated by theta_base.
            log(f"Candidate pool frozen before interventions: {len(candidates)} states")

            intervention_path = output_dir / "interventions.jsonl"
            continuation_path = output_dir / "continuations.jsonl.gz"
            completed_rows = read_jsonl(intervention_path)
            _validate_completed_accuracy_records(
                completed_rows,
                enabled=bool(config["accuracy_analysis"]["enabled"]),
            )
            completed = {int(row["intervention_id"]) for row in completed_rows}
            sanitize_continuation_audit(continuation_path, completed)
            pending = pending_candidates(candidates, completed)
            if completed:
                log(f"Resume: preserving {len(completed)} completed interventions")
            for candidate in pending:
                intervention_id = int(candidate["candidate_id"])
                restore_parameters_exact(student, base_snapshot)
                prefix = torch.tensor(candidate["prefix_token_ids"], dtype=torch.long, device=device)
                before_seeds, after_seeds = branch_seed_sets(
                    seed + 10_000_000 + intervention_id * 10_000,
                    int(config["intervention"]["num_continuations"]),
                )
                before_prefix = prefix.detach().clone()
                before_rollout = _generate_suffixes(
                    engine, student, before_prefix, before_seeds, config, tokenizer
                )
                update = _single_local_update(student, teacher, prefix, config)
                after_prefix = prefix.detach().clone()
                assert_same_prefix_independent_branches(
                    before_prefix, after_prefix, before_seeds, after_seeds
                )
                after_rollout = _generate_suffixes(
                    engine, student, after_prefix, after_seeds, config, tokenizer
                )

                # The evaluator is theta_base for both distributions.
                optimizer = update.pop("optimizer")
                del optimizer
                restore_parameters_exact(student, base_snapshot)
                gc.collect()
                torch.cuda.empty_cache()
                before_stats, before_audit, before_lengths = _score_suffixes_with_base(
                    student, teacher, before_rollout, before_seeds, "before", config,
                    tokenizer, candidate
                )
                after_stats, after_audit, after_lengths = _score_suffixes_with_base(
                    student, teacher, after_rollout, after_seeds, "after", config,
                    tokenizer, candidate
                )
                audit_rows = []
                for row in before_audit + after_audit:
                    row.update({
                        "intervention_id": intervention_id,
                        "candidate_id": intervention_id,
                        "dataset_index": candidate["dataset_index"],
                        "sample_id": candidate["sample_id"],
                        "quality_evaluator": "restored_theta_base_student_and_frozen_teacher",
                        "quality_support": "state_specific_student_top16_under_theta_base",
                    })
                    audit_rows.append(row)
                accuracy_fields = intervention_accuracy_fields(
                    [bool(row["predicted_correct"]) for row in before_audit],
                    [bool(row["predicted_correct"]) for row in after_audit],
                )
                accuracy_before = float(accuracy_fields["accuracy_before"])
                accuracy_after = float(accuracy_fields["accuracy_after"])
                pass_before = int(accuracy_fields["pass8_before"])
                pass_after = int(accuracy_fields["pass8_after"])
                append_gzip_jsonl(continuation_path, audit_rows)

                event = {
                    "intervention_id": intervention_id,
                    "candidate_id": intervention_id,
                    "dataset_index": candidate["dataset_index"],
                    "sample_id": candidate["sample_id"],
                    "state_position_t": candidate["state_position_t"],
                    "t": candidate["state_position_t"],
                    "prefix_length": candidate["prefix_length"],
                    "local_support_definition": "fixed_student_top16_at_theta_base",
                    "local_support_token_ids": update["support_ids"],
                    "local_p_base_conditional": update["p_base_conditional"],
                    "local_q_teacher_conditional": update["q_teacher_conditional"],
                    "local_reverse_kl_before": update["d0"],
                    "local_reverse_kl_after": update["d1"],
                    "D0": update["d0"],
                    "D1": update["d1"],
                    "delta_immediate": update["delta_immediate"],
                    "local_update_loss": update["loss"],
                    "gradient_norm": update["gradient_norm"],
                    "optimizer": {
                        "name": "AdamW", "learning_rate": float(config["training"]["learning_rate"]),
                        "betas": config["training"]["adam_betas"], "weight_decay": 0.0,
                        "max_grad_norm": float(config["training"]["max_grad_norm"]),
                        "steps": 1,
                    },
                    "before_future": before_stats,
                    "after_future": after_stats,
                    "future_quality_before_mean": before_stats["mean"],
                    "future_quality_after_mean": after_stats["mean"],
                    "future_before_mean": before_stats["mean"],
                    "future_before_std": before_stats["std_across_continuation_means"],
                    "future_after_mean": after_stats["mean"],
                    "future_after_std": after_stats["std_across_continuation_means"],
                    "delta_future": before_stats["mean"] - after_stats["mean"],
                    "continuation_lengths_before": before_lengths,
                    "continuation_lengths_after": after_lengths,
                    "before_seeds": list(before_seeds),
                    "after_seeds": list(after_seeds),
                    "same_exact_prefix": True,
                    "independent_branch_samples": True,
                    "base_restoration_verified_exact": True,
                    "future_evaluator": "theta_base",
                    "grad_norm": update["gradient_norm"],
                    "learning_rate": float(config["training"]["learning_rate"]),
                    "K": int(config["intervention"]["num_continuations"]),
                    "future_horizon": int(config["intervention"]["future_horizon"]),
                    "accuracy_max_new_tokens": int(
                        config["accuracy_analysis"]["max_new_tokens"]
                    ),
                    **accuracy_fields,
                    "generated_length_before_mean": fmean(before_lengths),
                    "generated_length_after_mean": fmean(after_lengths),
                    "experiment_schema_version": 2,
                    "accuracy_analysis_enabled": bool(config["accuracy_analysis"]["enabled"]),
                }
                validate_intervention_schema(event)
                append_jsonl(intervention_path, event)
                step = intervention_id + 1
                writer.add_scalar("immediate/D0", update["d0"], step)
                writer.add_scalar("immediate/D1", update["d1"], step)
                writer.add_scalar("immediate/gain", event["delta_immediate"], step)
                writer.add_scalar("update/grad_norm", update["gradient_norm"], step)
                writer.add_scalar("downstream/before_mean", before_stats["mean"], step)
                writer.add_scalar("downstream/after_mean", after_stats["mean"], step)
                writer.add_scalar("downstream/delta", event["delta_future"], step)
                writer.add_scalar("downstream/before_std", before_stats["std_across_continuation_means"], step)
                writer.add_scalar("downstream/after_std", after_stats["std_across_continuation_means"], step)
                writer.add_scalar("accuracy/before", accuracy_before, step)
                writer.add_scalar("accuracy/after", accuracy_after, step)
                writer.add_scalar("accuracy/delta", event["delta_accuracy"], step)
                writer.add_scalar("accuracy/pass8_before", pass_before, step)
                writer.add_scalar("accuracy/pass8_after", pass_after, step)
                writer.flush()
                log(
                    f"intervention={step}/{len(candidates)} "
                    f"delta_immediate={event['delta_immediate']:.6g} "
                    f"delta_future={event['delta_future']:.6g} restored_exact=true"
                )
            summary = analyze_output(output_dir)
            final_rows = read_jsonl(intervention_path)
            writer.add_histogram(
                "final/delta_immediate",
                torch.tensor([row["delta_immediate"] for row in final_rows]),
                len(final_rows),
            )
            writer.add_histogram(
                "final/delta_future",
                torch.tensor([row["delta_future"] for row in final_rows]),
                len(final_rows),
            )
            writer.add_histogram(
                "final/delta_accuracy",
                torch.tensor([row["delta_accuracy"] for row in final_rows]),
                len(final_rows),
            )
            writer.flush()
            if bool(config["accuracy_analysis"]["enabled"]):
                accuracy_summary, final_report = analyze_accuracy(
                    output_dir,
                    bootstrap_replicates=int(
                        config["accuracy_analysis"]["bootstrap_replicates"]
                    ),
                    bootstrap_seed=int(config["accuracy_analysis"]["bootstrap_seed"]),
                    zero_tolerance=summary.get("zero_tolerance"),
                )
                log("Future-KL/accuracy final report:\n" + final_report)
                if float(accuracy_summary["truncation_rate"]) > 0.01:
                    log(
                        "WARNING: more than 1% of accuracy continuations reached "
                        "ACCURACY_MAX_NEW_TOKENS; consider increasing the limit."
                    )
            log(f"Completed {summary['num_interventions']} interventions; Figure 1 written")
        except BaseException:
            if read_jsonl(output_dir / "interventions.jsonl"):
                completed_now = {
                    int(row["intervention_id"])
                    for row in read_jsonl(output_dir / "interventions.jsonl")
                }
                sanitize_continuation_audit(
                    output_dir / "continuations.jsonl.gz", completed_now
                )
                analyze_output(output_dir)
            raise
        finally:
            writer.close()
            engine.close()
            del student, teacher, tokenizer
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-states", type=int)
    parser.add_argument("--k-rollouts", type=int)
    parser.add_argument("--future-horizon", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--accuracy-max-new-tokens", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    accuracy_group = parser.add_mutually_exclusive_group()
    accuracy_group.add_argument(
        "--enable-accuracy-analysis", dest="enable_accuracy_analysis",
        action="store_true", default=None
    )
    accuracy_group.add_argument(
        "--disable-accuracy-analysis", dest="enable_accuracy_analysis",
        action="store_false"
    )
    args = parser.parse_args()
    output = run(args)
    print(f"Completed experiment: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
