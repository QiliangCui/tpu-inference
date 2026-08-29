#!/bin/bash
# Measure current mla/v2 kernel (mla_ragged_paged_attention) performance at
# the params it ships with (tuned_params_mapping). No tuning, no sweeping —
# one latency number per batched_decode config.
#
# Requirements: TPU host (v6e+), an env where jax + tpu_inference import
# (e.g. `conda activate vllm`), TPU devices free (kill any running vllm).
#
# Usage (from anywhere; ~10 min full run, compile-dominated):
#   scripts/bench_mla.sh                              # all configs
#   scripts/bench_mla.sh --filter tokens_16           # subset by case name
#   scripts/bench_mla.sh --output /tmp/mla.json       # bare JSON to a file
#   scripts/bench_mla.sh --iters 50 --warmup 5 --batches 3   # timing knobs
#
# Output: progress on stderr; on stdout a JSON list wrapped in
# ==RESULT START== / ==RESULT END== markers, one entry per config:
#   {"input": {tokens, heads, seqs, ...}, "tuned": {BS, pages, ...},
#    "latency_us": 50.42}   (or "error": "..." on failure)
# Feed the log to scripts/evaluate_mla.sh for the average + reward.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="${PYTHONPATH:-}:." python scripts/benchmarking/kernels/benchmark_mla_v2.py "$@"
