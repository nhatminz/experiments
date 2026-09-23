# CMT exact-prefix state-intervention experiment

This folder answers one question:

> After training on a state `s_t`, does the distribution/quality of subsequent
> states visited by the student change, even when immediate gains are matched?

It is a causal intervention runner, not the observational CMT motivation
summary and not full CMT training. `BellmanOPD_analysis` is imported read-only.
The runner is intentionally single-GPU: set `CUDA_VISIBLE_DEVICES` to exactly
one device. The colocated HF student/teacher and TP=1 vLLM replica share it in
the same way as the main training rollout pipeline.

For every event, the runner generates an on-policy trajectory, computes the
existing CMT diagnostics, selects one eligible `s_t` uniformly without looking
at `D_t`, probes suffixes from the exact prefix, performs one Student-Top-K OPD
update at only that state (weight 1, no Gibbs), and probes again from the same
prefix with the same seeds.

Outputs are written under `outputs/<run>/`:

- `events.jsonl`: one intervention per row, including the exact `s_t` token
  prefix and fixed pre-update Student-Top-K/teacher target used locally;
- `future_states.jsonl.gz`: per future state, before/after;
- `resolved_config.yaml`, `run_metadata.json`;
- `matched_g_bins.csv`, `summary.json`.

## Two-step smoke run

```bash
cd /workspace/storage-shared/nlp/minhpn19/Experiment
MAIN_REPO=/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis \
CUDA_VISIBLE_DEVICES=0 MAX_STEPS=2 NUM_PROBE_ROLLOUTS=4 FUTURE_HORIZON=256 \
  bash scripts/run.sh
```

For a cheaper plumbing-only smoke test, set `NUM_PROBE_ROLLOUTS=2` and
`FUTURE_HORIZON=32`.

## Normal 100-intervention run

```bash
cd /workspace/storage-shared/nlp/minhpn19/Experiment
MAIN_REPO=/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis \
CUDA_VISIBLE_DEVICES=0 MAX_STEPS=100 NUM_PROBE_ROLLOUTS=4 FUTURE_HORIZON=256 \
TOP_K=16 LEARNING_RATE=5e-6 SEED=42 \
  bash scripts/run.sh
```

Important optional overrides are `STUDENT_MODEL`, `TEACHER_MODEL`,
`TRAIN_DATA`, `PROMPT_KEY`, `TRAIN_DATA_SPLIT`, `STORAGE_ROOT`,
`NORMAL_ROLLOUT_HORIZON`, `MIN_ORIGINAL_SUFFIX_TOKENS`,
`ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION`, and `ROLLOUT_VLLM_MAX_MODEL_LEN`.

Re-run only the matched-g analysis without inference:

```bash
python3 analyze.py outputs/<run> --bins 4
```
