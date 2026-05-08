#!/bin/bash
# Run a small benchmark with Mosaic/HLO dump enabled, then upload results.
# Try multiple dump approaches since we don't know which flag works.
set -euo pipefail

DUMP_DIR="/tmp/mosaic_dump"
XLA_DUMP_DIR="/tmp/xla_dump"
mkdir -p "$DUMP_DIR" "$XLA_DUMP_DIR"

# Try both mosaic dump and standard xla dump
export XLA_FLAGS="${XLA_FLAGS:-} --xla_mosaic_dump_to=$DUMP_DIR --xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text"
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

# === Upload to GCS ===
echo ""
echo "=== Uploading to GCS ==="
python3 << 'PYEOF'
from google.cloud import storage
import os, glob

client = storage.Client()
bucket = client.bucket('vllm-cb-storage2')

# Upload mosaic dump files
for dump_dir, prefix in [('/tmp/mosaic_dump', 'mosaic_dump'), ('/tmp/xla_dump', 'xla_dump')]:
    files = sorted(glob.glob(dump_dir + '/**', recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    # Filter for fused-moe related files, or take first 10
    fused_files = [f for f in files if 'fused' in f.lower() or 'moe' in f.lower()]
    upload_files = fused_files[:10] if fused_files else files[:10]

    for f in upload_files:
        rel_path = os.path.relpath(f, dump_dir)
        blob_path = f'cuiq/{prefix}/{rel_path}'
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(f)
        print(f'  Uploaded {rel_path} ({os.path.getsize(f)/1024:.0f} KB)')

    if not upload_files:
        print(f'  No files to upload from {dump_dir}')

print('Done.')
PYEOF
