#!/bin/bash
# Run a small benchmark with Mosaic IR dump enabled, then upload results.
set -euo pipefail

DUMP_DIR="/tmp/mosaic_dump"
mkdir -p "$DUMP_DIR"

export XLA_FLAGS="${XLA_FLAGS:-} --xla_mosaic_dump_to=$DUMP_DIR"
echo "=== XLA_FLAGS: $XLA_FLAGS ==="

# Run the normal benchmark
bash /workspace/tpu_inference/tests/e2e/benchmarking/sweep_profile.sh || true

# Report what was dumped
echo "=== Mosaic dump files ==="
find "$DUMP_DIR" -type f 2>/dev/null | head -50
echo ""
echo "=== Dump directory size ==="
du -sh "$DUMP_DIR" 2>/dev/null || echo "No dump dir"
echo ""
echo "=== fused-moe dump files ==="
find "$DUMP_DIR" -name "*fused-moe*" -o -name "*fused_moe*" 2>/dev/null | head -20
echo ""

# Show first fused-moe file (first 300 lines)
FIRST_FILE=$(find "$DUMP_DIR" -name "*fused-moe*" -type f 2>/dev/null | sort | head -1)
if [ -n "$FIRST_FILE" ]; then
    echo "=== First fused-moe file: $FIRST_FILE ==="
    echo "=== Size: $(du -h "$FIRST_FILE" | cut -f1) ==="
    head -300 "$FIRST_FILE"
    echo ""
    echo "=== (truncated at 300 lines) ==="
else
    echo "=== No fused-moe dump files found ==="
    echo "=== All dump files ==="
    ls -la "$DUMP_DIR"/ 2>/dev/null | head -30
    # Show first file regardless of name
    ANY_FILE=$(find "$DUMP_DIR" -type f 2>/dev/null | head -1)
    if [ -n "$ANY_FILE" ]; then
        echo "=== First file: $ANY_FILE ==="
        head -100 "$ANY_FILE"
    fi
fi

# Upload fused-moe dump files to GCS
python3 -c "
from google.cloud import storage
import os, glob
client = storage.Client()
bucket = client.bucket('vllm-cb-storage2')
dump_dir = '$DUMP_DIR'
files = sorted(glob.glob(dump_dir + '/*fused-moe*') + glob.glob(dump_dir + '/*fused_moe*'))
if not files:
    files = sorted(glob.glob(dump_dir + '/*'))[:10]
    print(f'No fused-moe files. Uploading first {len(files)} files instead.')
for f in files[:10]:
    name = os.path.basename(f)
    blob = bucket.blob(f'cuiq/mosaic_dump/{name}')
    blob.upload_from_filename(f)
    print(f'Uploaded {name} ({os.path.getsize(f)/1024:.0f} KB)')
print('Done.')
" 2>&1 || echo "GCS upload failed"
