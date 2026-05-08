#!/bin/bash
# Compile the model with HLO dump, extract fused_moe sections, upload to GCS.
# No benchmark — just compile and dump.

DUMP_DIR="/tmp/xla_dump"
EXTRACT="/tmp/fused_moe_hlo_extract.txt"
mkdir -p "$DUMP_DIR"

export XLA_FLAGS="--xla_dump_to=$DUMP_DIR --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text"
echo "XLA_FLAGS: $XLA_FLAGS"

# Start server, wait for compilation, then kill it
echo "Starting server for compilation only..."
MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=1 vllm serve \
  "${TEST_MODEL}" \
  --download-dir=/tmp/hf_home \
  --tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  --max-model-len="${MAX_MODEL_LEN}" \
  --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs="${MAX_NUM_SEQS}" \
  --enable-expert-parallel \
  --kv-cache-dtype="${KV_CACHE_DTYPE}" \
  --gpu-memory-utilization="${GPU_MEMORY_UTILIZATION}" \
  --seed=42 --no-enable-prefix-caching --async-scheduling \
  > /tmp/server.log 2>&1 &
SERVER_PID=$!

# Wait for server ready or timeout (30 min for 480B compilation)
echo "Waiting for compilation (up to 30 min)..."
READY=0
for i in $(seq 1 360); do
    if grep -q "Application startup complete" /tmp/server.log 2>/dev/null; then
        READY=1
        echo "Server ready after $((i*5))s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died during compilation"
        tail -20 /tmp/server.log
        break
    fi
    sleep 5
done

# Kill server — we only needed the compilation
kill -9 $SERVER_PID 2>/dev/null || true
sleep 2

# Print compilation log lines with fused_ep_moe
echo ""
echo "=== fused_ep_moe compilation log ==="
grep "fused_ep_moe" /tmp/server.log || echo "(no fused_ep_moe lines)"

# Report dump
echo ""
echo "=== XLA dump ==="
XLA_COUNT=$(find "$DUMP_DIR" -type f 2>/dev/null | wc -l)
echo "Total files: $XLA_COUNT"
du -sh "$DUMP_DIR" 2>/dev/null || true

# Extract fused_moe sections using grep
echo ""
echo "=== Extracting fused_moe from HLO ==="
> "$EXTRACT"

# Use a jit_step_fun file (contains fused_moe). Pick before_optimizations (less processed).
SMALLEST=$(find "$DUMP_DIR" -name "*jit_step_fun*before_optimizations.txt" -type f -printf '%s %p\n' 2>/dev/null | sort -n | head -1 | awk '{print $2}')
if [ -z "$SMALLEST" ]; then
    # Fallback: any jit_step_fun file
    SMALLEST=$(find "$DUMP_DIR" -name "*jit_step_fun*" -type f -printf '%s %p\n' 2>/dev/null | sort -n | head -1 | awk '{print $2}')
fi
if [ -z "$SMALLEST" ]; then
    # Fallback: largest file (likely the main module)
    SMALLEST=$(find "$DUMP_DIR" -type f -name "*.txt" -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
fi

if [ -n "$SMALLEST" ]; then
    FSIZE=$(du -h "$SMALLEST" | cut -f1)
    echo "File: $(basename $SMALLEST) ($FSIZE)" | tee -a "$EXTRACT"

    FCOUNT=$(grep -c "fused.moe\|fused_moe\|fused-moe" "$SMALLEST" 2>/dev/null || echo 0)
    echo "fused_moe references: $FCOUNT" | tee -a "$EXTRACT"
    echo "" >> "$EXTRACT"

    echo "--- custom_call_targets ---" >> "$EXTRACT"
    grep -o 'custom_call_target="[^"]*"' "$SMALLEST" 2>/dev/null | sort -u >> "$EXTRACT" 2>/dev/null || true
    echo "" >> "$EXTRACT"

    echo "--- fused_moe context (first 300 lines of matches) ---" >> "$EXTRACT"
    grep -n -B 2 -A 10 "fused.moe\|fused_moe\|fused-moe" "$SMALLEST" 2>/dev/null | head -300 >> "$EXTRACT" 2>/dev/null || true

    echo ""
    echo "Extract size: $(du -h "$EXTRACT" | cut -f1)"
    cat "$EXTRACT"
else
    echo "No dump files found"
fi

# Upload
echo ""
echo "=== Upload ==="
python3 -c "
from google.cloud import storage
import os
for f in ['/tmp/fused_moe_hlo_extract.txt', '/tmp/server.log']:
    if os.path.exists(f) and os.path.getsize(f) > 0:
        client = storage.Client()
        bucket = client.bucket('vllm-cb-storage2')
        name = os.path.basename(f)
        blob = bucket.blob(f'cuiq/mosaic_dump/{name}')
        blob.upload_from_filename(f)
        print(f'Uploaded {name} ({os.path.getsize(f)/1024:.0f} KB)')
" 2>&1 || echo "Upload failed"
echo "Done."
