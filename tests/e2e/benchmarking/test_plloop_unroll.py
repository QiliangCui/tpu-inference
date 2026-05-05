#!/usr/bin/env python3
"""Minimal test: pl.loop with unroll + DMA ops on TPU.

Tests whether partial unrolling (unroll=N where 1 < N < total) works
with async DMA operations inside a Pallas kernel on TPU.

Run: python3 test_plloop_unroll.py
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

NUM_ITEMS = 16


def test_variant(name, unroll_val):
    """Test a specific unroll value."""
    print(f"\n=== Test: {name} (unroll={unroll_val}) ===")

    def kernel(src_hbm, dst_hbm, acc_smem, sem):
        # Accumulate in SMEM, DMA each item from src to dst
        acc_smem[0] = 0

        @pl.loop(0, NUM_ITEMS, unroll=unroll_val)
        def body(i):
            acc_smem[0] = acc_smem[0] + src_hbm[i]
            pltpu.make_async_copy(
                src_ref=src_hbm.at[pl.ds(i, 1)],
                dst_ref=dst_hbm.at[pl.ds(i, 1)],
                sem=sem,
            ).start()

        # Wait for all DMAs
        pltpu.make_async_copy(
            src_ref=dst_hbm.at[pl.ds(0, NUM_ITEMS)],
            dst_ref=dst_hbm.at[pl.ds(0, NUM_ITEMS)],
            sem=sem,
        ).wait()

    try:
        f = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((NUM_ITEMS,), jnp.float32),
            grid_spec=pl.GridSpec(
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.ANY),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
                scratch_shapes=[
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SemaphoreType.DMA,
                ],
            ),
        )
        src = jnp.arange(NUM_ITEMS, dtype=jnp.float32)
        result = f(src)
        print(f"  PASSED! Result: {result[:4]}...")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print(f"JAX devices: {jax.device_count()}")
    print(f"Platform: {jax.default_backend()}")

    test_variant("no unroll", False)
    test_variant("unroll=2", 2)
    test_variant("unroll=4", 4)
    test_variant("unroll=8", 8)
    test_variant("full unroll", True)

    print("\n=== Also test lax.fori_loop variants ===")

    def test_fori(name, unroll_val):
        print(f"\n=== Test: {name} (fori_loop unroll={unroll_val}) ===")

        def kernel(src_hbm, dst_hbm, acc_smem, sem):
            def body(i, acc):
                pltpu.make_async_copy(
                    src_ref=src_hbm.at[pl.ds(i, 1)],
                    dst_ref=dst_hbm.at[pl.ds(i, 1)],
                    sem=sem,
                ).start()
                return acc + 1

            lax.fori_loop(0, NUM_ITEMS, body, 0, unroll=unroll_val)

            pltpu.make_async_copy(
                src_ref=dst_hbm.at[pl.ds(0, NUM_ITEMS)],
                dst_ref=dst_hbm.at[pl.ds(0, NUM_ITEMS)],
                sem=sem,
            ).wait()

        try:
            f = pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((NUM_ITEMS,), jnp.float32),
                grid_spec=pl.GridSpec(
                    in_specs=[pl.BlockSpec(memory_space=pltpu.ANY)],
                    out_specs=pl.BlockSpec(memory_space=pltpu.ANY),
                    scratch_shapes=[
                        pltpu.SMEM((1,), jnp.int32),
                        pltpu.SemaphoreType.DMA,
                    ],
                ),
            )
            src = jnp.arange(NUM_ITEMS, dtype=jnp.float32)
            result = f(src)
            print(f"  PASSED! Result: {result[:4]}...")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    test_fori("fori no unroll", False)
    test_fori("fori unroll=4", 4)
    test_fori("fori unroll=8", 8)
    test_fori("fori full unroll", True)
