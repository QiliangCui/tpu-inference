#!/usr/bin/env python3
"""Decode benchmark: isolates decode performance using prefix caching.

Approach:
  1. Seed KV cache by generating max_tokens=1 for each prompt
  2. Re-run same prompts with max_tokens=output_len
  3. Prefix caching reuses KV from step 1 -> measures pure decode

This avoids the benchmark_serving pitfall where throughput mixes
prefill and decode phases (build #272 lesson).

Usage:
  USE_MOE_EP_KERNEL=0 python3 decode_bench.py \
    --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --download-dir /tmp/hf_home \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --max-model-len 10240 --max-num-batched-tokens 8192 \
    --max-num-seqs 512 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 --seed 42
"""

import csv
from datetime import datetime
import os
import statistics
import time

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

NUM_ITERS = 5  # More iters for stable median (first may be compilation)
OUTPUT_LEN = 256  # Enough decode steps to amortize overhead

# Scenarios for 480B Qwen3-Coder EP comparison.
# Batch sizes chosen based on observed concurrency in prior experiments:
#   - 8K context: ~32 concurrent sequences fit (builds #272, #273)
#   - 1K context: ~64+ fit easily
SCENARIOS = [
    {
        "name": "decode_8k",
        "in": 8192,
        "out": OUTPUT_LEN,
        "batches": [1, 4, 8, 16, 32],
        "mode": "decode",
    },
    {
        "name": "decode_1k",
        "in": 1024,
        "out": OUTPUT_LEN,
        "batches": [1, 8, 16, 32, 64],
        "mode": "decode",
    },
]

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_TRACK = (
    "fused_ep"
    if os.environ.get("USE_MOE_EP_KERNEL", "0") == "1"
    else "gmm_ep"
)
OUTPUT_CSV = f"/tmp/decode_bench_{_TRACK}_{_TIMESTAMP}.csv"


def init_csv():
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track",
            "scenario",
            "input_len",
            "output_len",
            "batch_size",
            "tpot_ms",
            "tps_total",
            "tps_per_chip",
            "iter_times",
            "status",
        ])


def log_result(
    track, scenario, input_len, output_len, bs, tpot, tps_total, tps_chip,
    iter_times, status,
):
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            track, scenario, input_len, output_len, bs,
            round(tpot, 2), round(tps_total, 1), round(tps_chip, 1),
            ";".join(f"{t:.3f}" for t in iter_times), status,
        ])


def run_bench(args):
    init_csv()

    engine_args = EngineArgs.from_cli_args(args)

    # Enable prefix caching — required for decode isolation trick.
    # The second generate() call with same prompts hits KV cache,
    # so only decode is measured.
    engine_args.enable_prefix_caching = True
    engine_args.disable_log_stats = True

    tp = engine_args.tensor_parallel_size
    track = _TRACK

    print(f"=== Decode Benchmark — Track: {track}, TP={tp} ===")
    print(f"Model: {engine_args.model}")
    print(f"USE_MOE_EP_KERNEL={os.environ.get('USE_MOE_EP_KERNEL', '0')}")
    print(f"enable_prefix_caching=True (for decode isolation)")
    print(f"NUM_ITERS={NUM_ITERS}, OUTPUT_LEN={OUTPUT_LEN}")
    print()

    llm = LLM.from_engine_args(engine_args)

    # Warmup: cover both ISLs to trigger compilation
    print(">>> Warmup (8K + 1K)...", end=" ", flush=True)
    for isl in [8192, 1024]:
        warmup = [{"prompt_token_ids": [100] * isl}]
        llm.generate(
            warmup,
            SamplingParams(max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
    print("done\n")

    results = []

    for sc in SCENARIOS:
        print(
            f">>> Scenario: {sc['name']} "
            f"(ISL={sc['in']}, OSL={sc['out']})"
        )
        print(
            f"    {'BS':>4s} | {'TPOT(ms)':>10s} | {'TPS':>8s} | "
            f"{'TPS/chip':>10s} | {'status':>6s}"
        )
        print(f"    {'-'*55}")

        for bs in sc["batches"]:
            # Create unique prompts for this batch.
            # Each prompt has distinct tokens so prefix caching
            # doesn't cross-contaminate between batch sizes.
            prompts = [
                {"prompt_token_ids": [1000 + bs * 100 + i] * sc["in"]}
                for i in range(bs)
            ]

            try:
                # Phase 1: Seed KV cache (prefill only)
                llm.generate(
                    prompts,
                    SamplingParams(max_tokens=1, min_tokens=1, ignore_eos=True),
                    use_tqdm=False,
                )

                # Phase 2: Decode benchmark (prefix cache hit -> pure decode)
                decode_params = SamplingParams(
                    max_tokens=sc["out"],
                    min_tokens=sc["out"],
                    ignore_eos=True,
                )
                iter_times = []

                for _ in range(NUM_ITERS):
                    start = time.perf_counter()
                    llm.generate(prompts, decode_params, use_tqdm=False)
                    elapsed = time.perf_counter() - start
                    iter_times.append(elapsed)

                # TPOT = time per decode step (all seqs advance together)
                tpot_list = [(t / sc["out"]) * 1000 for t in iter_times]
                tps_list = [(bs * sc["out"]) / t for t in iter_times]

                med_tpot = statistics.median(tpot_list)
                med_tps = statistics.median(tps_list)
                tps_chip = med_tps / tp

                print(
                    f"    {bs:4d} | {med_tpot:10.2f} | {med_tps:8.0f} | "
                    f"{tps_chip:10.0f} | PASS"
                )
                log_result(
                    track, sc["name"], sc["in"], sc["out"], bs,
                    med_tpot, med_tps, tps_chip, iter_times, "PASS",
                )
                results.append({
                    "scenario": sc["name"], "bs": bs,
                    "tpot": med_tpot, "tps": med_tps, "tps_chip": tps_chip,
                })

            except Exception as e:
                err = str(e)[:80]
                print(f"    {bs:4d} | {'--':>10s} | {'--':>8s} | {'--':>10s} | FAIL")
                print(f"           Error: {err}")
                log_result(
                    track, sc["name"], sc["in"], sc["out"], bs,
                    0, 0, 0, [], f"FAIL:{err}",
                )
                # If OOM, skip larger batch sizes for this scenario
                if "out of memory" in str(e).lower() or "resource" in str(e).lower():
                    print("           Skipping larger batch sizes (OOM)")
                    break
                continue

        print()

    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY — {track}")
    print(f"{'='*70}")
    if results:
        print(f"{'scenario':<12s} {'BS':>4s} {'TPOT(ms)':>10s} {'TPS':>8s} {'TPS/chip':>10s}")
        for r in results:
            print(
                f"{r['scenario']:<12s} {r['bs']:4d} "
                f"{r['tpot']:10.2f} {r['tps']:8.0f} {r['tps_chip']:10.0f}"
            )
    print(f"\nCSV: {OUTPUT_CSV}")

    # Upload CSV to GCS if bucket is configured
    gcs_bucket = os.environ.get("GCS_BUCKET", "gs://vllm-cb-storage2")
    try:
        from google.cloud import storage as gcs

        bucket_name = gcs_bucket.replace("gs://", "").split("/")[0]
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob_path = f"cuiq/sweep/decode_bench/{os.path.basename(OUTPUT_CSV)}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(OUTPUT_CSV)
        print(f"Uploaded to gs://{bucket_name}/{blob_path}")
    except Exception as e:
        print(f"GCS upload skipped: {e}")


def parse_args():
    parser = FlexibleArgumentParser(
        description="Decode benchmark with prefix caching isolation"
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bench(args)
