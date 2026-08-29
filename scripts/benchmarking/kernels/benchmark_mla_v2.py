# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Report the current latency of the mla/v2 kernel at its shipped params.

For each batched_decode entry in tuned_params_mapping, runs
mla_ragged_paged_attention exactly as it ships (the tuned params the serving
path would pick) and reports the measured per-call latency. No tuning, no
sweeping, no baseline comparison — just current performance.

Output: a JSON list on stdout, one entry per case:
    [{"input": {...}, "tuned": {...}, "latency_us": 50.42}, ...]
Progress goes to stderr so stdout stays valid JSON.

Run from the repo root on a TPU host:
    python scripts/benchmarking/kernels/benchmark_mla_v2.py
    python scripts/benchmarking/kernels/benchmark_mla_v2.py --filter tokens_16
"""

import argparse
import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from tests.kernels.mla_v2_test import generate_mla_inputs
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention
from tpu_inference.kernels.mla.v2.tuned_params import tuned_params_mapping


def measure_us(fn, *, iters, warmup, batches):
    """Median-of-batches wall-clock latency (us) per call, device-synced."""
    out = None
    for _ in range(warmup):
        out = fn()
    jax.block_until_ready(out)
    per_batch = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iters):
            out = fn()
        jax.block_until_ready(out)
        per_batch.append((time.perf_counter_ns() - start) / iters)
    return statistics.median(per_batch) / 1e3


def key_name(key):
    return (f"tokens_{key.max_num_tokens}_heads_{key.actual_num_q_heads}_"
            f"seqs_{key.max_num_seqs}_pagesperseq_{key.pages_per_seq}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter",
                        default="",
                        help="only run cases whose name contains this string")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches",
                        type=int,
                        default=3,
                        help="timing batches per case (median taken)")
    parser.add_argument("--output",
                        default=None,
                        help="write the JSON here instead of stdout (stdout "
                        "also carries vllm import logs)")
    args = parser.parse_args()

    assert jax.devices()[0].platform == "tpu", "requires a TPU host"
    print(
        f"device={jax.devices()[0].device_kind}, "
        f"cases=batched_decode entries of tuned_params_mapping, "
        f"iters={args.iters} x {args.batches} batches (median)",
        file=sys.stderr)

    results = []
    for key, params in tuned_params_mapping.items():
        if key.case != "batched_decode":
            continue
        name = key_name(key)
        if args.filter and args.filter not in name:
            continue

        kv_len = key.pages_per_seq * key.page_size_per_kv_packing * key.kv_packing
        record = {
            "input": {
                "tokens": key.max_num_tokens,
                "heads": key.actual_num_q_heads,
                "seqs": key.max_num_seqs,
                "pages_per_seq": key.pages_per_seq,
                "kv_len": kv_len,
                "lkv_dim": key.actual_lkv_dim,
                "r_dim": key.actual_r_dim,
                "q_dtype": key.q_dtype,
                "kv_dtype": key.kv_dtype,
            },
            "tuned": {
                "BS": params.decode_batch_size,
                "pages": params.num_kv_pages_per_block,
                "queries_per_block": params.num_queries_per_block,
                "vmem_limit_bytes": params.vmem_limit_bytes,
            },
        }
        print(f"running {name}...", file=sys.stderr)

        rng = np.random.default_rng(1234)
        inputs = generate_mla_inputs(
            seq_lens=[[1, kv_len] for _ in range(key.max_num_seqs)],
            num_heads=key.actual_num_q_heads,
            lkv_dim=key.actual_lkv_dim,
            r_dim=key.actual_r_dim,
            page_size=key.page_size_per_kv_packing * key.kv_packing,
            q_dtype=jnp.dtype(key.q_dtype),
            kv_dtype=jnp.dtype(key.kv_dtype),
            num_pages=key.pages_per_seq * key.max_num_seqs,
            rng=rng,
        )
        (ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices,
         cu_q_lens, distribution) = inputs
        ql_nope = jnp.transpose(ql_nope, (1, 0, 2))

        # The kernel donates cache_kv (in-place KV update); chain the returned
        # cache into the next call so repeated timing calls stay valid.
        state = {"cache": cache_kv}

        def run():
            out, state["cache"] = mla_ragged_paged_attention(
                ql_nope=ql_nope,
                q_pe=q_pe,
                new_kv_c=new_kv_c,
                new_k_pe=new_k_pe,
                cache_kv=state["cache"],
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                sliding_window=key.sliding_window,
                soft_cap=key.soft_cap,
                q_scale=None,
                k_scale=None,
                v_scale=None,
                chunk_prefill_size=key.chunk_prefill_size,
                s_dtype=key.s_dtype,
                p_same_dtype_as_v=key.p_same_dtype_as_v,
                decode_batch_size=params.decode_batch_size,
                num_kv_pages_per_block=params.num_kv_pages_per_block,
                num_queries_per_block=params.num_queries_per_block,
                vmem_limit_bytes=params.vmem_limit_bytes,
            )
            return out

        try:
            latency = measure_us(run,
                                 iters=args.iters,
                                 warmup=args.warmup,
                                 batches=args.batches)
            record["latency_us"] = round(latency, 2)
        except Exception as e:
            record["error"] = " ".join(str(e).split())[:200]
        results.append(record)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"wrote {len(results)} results to {args.output}",
              file=sys.stderr)
    else:
        print("==RESULT START==")
        json.dump(results, sys.stdout, indent=2)
        print("\n==RESULT END==")


if __name__ == "__main__":
    main()
