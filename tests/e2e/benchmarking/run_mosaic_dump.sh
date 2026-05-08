#!/bin/bash
# Run a small benchmark with Mosaic/HLO dump enabled, then upload results.
# Try multiple dump approaches since we don't know which flag works.
set -euo pipefail

DUMP_DIR="/tmp/mosaic_dump"
XLA_DUMP_DIR="/tmp/xla_dump"
mkdir -p "$DUMP_DIR" "$XLA_DUMP_DIR"

# Try both mosaic dump and standard xla dump
# Note: --xla_mosaic_dump_to produced no output (build #345).
# Both flags together caused JAX to find 0 devices (build #347).
# Try only --xla_dump_to (standard XLA, more likely to work).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text"
echo "=== XLA_FLAGS: $XLA_FLAGS ==="

# Run the normal benchmark
bash /workspace/tpu_inference/tests/e2e/benchmarking/sweep_profile.sh || true

# === Report Mosaic dump ===
echo ""
echo "=========================================="
echo "=== Mosaic dump ($DUMP_DIR) ==="
echo "=========================================="
MOSAIC_COUNT=$(find "$DUMP_DIR" -type f 2>/dev/null | wc -l)
echo "Total files: $MOSAIC_COUNT"
du -sh "$DUMP_DIR" 2>/dev/null
if [ "$MOSAIC_COUNT" -gt 0 ]; then
    echo "=== File listing ==="
    find "$DUMP_DIR" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -20
    echo ""
    echo "=== fused-moe files ==="
    find "$DUMP_DIR" -name "*fused*" -type f 2>/dev/null | head -10
fi

# === Report XLA dump ===
echo ""
echo "=========================================="
echo "=== XLA dump ($XLA_DUMP_DIR) ==="
echo "=========================================="
XLA_COUNT=$(find "$XLA_DUMP_DIR" -type f 2>/dev/null | wc -l)
echo "Total files: $XLA_COUNT"
du -sh "$XLA_DUMP_DIR" 2>/dev/null
if [ "$XLA_COUNT" -gt 0 ]; then
    echo "=== File listing (largest first) ==="
    find "$XLA_DUMP_DIR" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -20
    echo ""
    # Find any file mentioning fused_moe or custom-call
    echo "=== Files containing fused_moe ==="
    grep -rl "fused.moe\|fused_moe\|fused-moe" "$XLA_DUMP_DIR" 2>/dev/null | head -5
    echo ""
    # Show a snippet of a custom-call (fused_ep_moe shows up as custom-call in HLO)
    FUSED_FILE=$(grep -rl "fused.moe\|fused_moe\|fused-moe" "$XLA_DUMP_DIR" 2>/dev/null | head -1)
    if [ -n "$FUSED_FILE" ]; then
        echo "=== Snippet from: $FUSED_FILE ==="
        grep -A 5 -B 2 "fused.moe\|fused_moe\|fused-moe" "$FUSED_FILE" | head -50
    fi
fi

# === Extract fused_moe custom-call from HLO files using grep (memory-efficient) ===
echo ""
echo "=== Extracting fused_moe references ==="

EXTRACT="/tmp/fused_moe_hlo_extract.txt"
> "$EXTRACT"

# Find the smallest before_optimizations file (least processed, easier to read)
SMALLEST=$(find "$XLA_DUMP_DIR" -name "*before_optimizations*" -type f -printf '%s %p\n' 2>/dev/null | sort -n | head -1 | awk '{print $2}')
if [ -z "$SMALLEST" ]; then
    SMALLEST=$(find "$XLA_DUMP_DIR" -name "*.txt" -type f -printf '%s %p\n' 2>/dev/null | sort -n | head -1 | awk '{print $2}')
fi

if [ -n "$SMALLEST" ]; then
    echo "Using: $(basename $SMALLEST) ($(du -h "$SMALLEST" | cut -f1))"
    echo ""

    # Count fused_moe references
    FUSED_COUNT=$(grep -c "fused.moe\|fused_moe\|fused-moe" "$SMALLEST" 2>/dev/null || echo 0)
    echo "fused_moe references: $FUSED_COUNT"

    # Extract lines with context
    echo "=== FILE: $(basename $SMALLEST) ===" >> "$EXTRACT"
    echo "fused_moe references: $FUSED_COUNT" >> "$EXTRACT"
    echo "" >> "$EXTRACT"
    grep -n -B 3 -A 15 "fused.moe\|fused_moe\|fused-moe" "$SMALLEST" 2>/dev/null | head -500 >> "$EXTRACT"

    echo "" >> "$EXTRACT"
    echo "=== custom-call targets ===" >> "$EXTRACT"
    grep -o 'custom_call_target="[^"]*"' "$SMALLEST" 2>/dev/null | sort -u >> "$EXTRACT"

    echo "" >> "$EXTRACT"
    echo "=== backend_config with fused ===" >> "$EXTRACT"
    grep -B 1 -A 5 'backend_config.*fused\|fused.*backend_config' "$SMALLEST" 2>/dev/null | head -200 >> "$EXTRACT"

    echo ""
    echo "Extract size: $(du -h "$EXTRACT" | cut -f1)"
    echo ""
    echo "=== Extract content (first 3000 chars) ==="
    head -c 3000 "$EXTRACT"
    echo ""
    echo "=== (truncated) ==="
else
    echo "No HLO dump files found"
fi

# Upload extract to GCS
python3 -c "
from google.cloud import storage
import os
f = '/tmp/fused_moe_hlo_extract.txt'
if os.path.exists(f) and os.path.getsize(f) > 0:
    client = storage.Client()
    bucket = client.bucket('vllm-cb-storage2')
    blob = bucket.blob('cuiq/mosaic_dump/fused_moe_hlo_extract.txt')
    blob.upload_from_filename(f)
    print(f'Uploaded ({os.path.getsize(f)/1024:.0f} KB)')
else:
    print('Nothing to upload')
" 2>&1 || echo "GCS upload failed"
