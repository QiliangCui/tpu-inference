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

# === Extract fused_moe custom-call from smallest HLO file ===
echo ""
echo "=== Extracting fused_moe custom-call body ==="

python3 << 'PYEOF'
import os, glob, re

dump_dir = '/tmp/xla_dump'
out_file = '/tmp/fused_moe_hlo_extract.txt'

# Find the smallest file that contains fused_moe (before_optimizations is usually smaller)
candidates = []
for f in sorted(glob.glob(dump_dir + '/*before_optimizations*')):
    if os.path.isfile(f):
        candidates.append((os.path.getsize(f), f))

if not candidates:
    for f in sorted(glob.glob(dump_dir + '/*.txt')):
        if os.path.isfile(f):
            candidates.append((os.path.getsize(f), f))

candidates.sort()
print(f"Found {len(candidates)} candidate files")

with open(out_file, 'w') as out:
    for size, filepath in candidates[:3]:  # Check smallest 3
        print(f"Checking {os.path.basename(filepath)} ({size/1024/1024:.0f} MB)...")
        with open(filepath) as f:
            content = f.read()

        # Find all custom-call ops that mention fused_moe
        # HLO custom-calls look like: %custom-call = ... custom-call(...), custom_call_target="mosaic_..."
        # The kernel name appears in the backend_config
        fused_sections = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'fused-moe' in line or 'fused_moe' in line:
                start = max(0, i - 5)
                end = min(len(lines), i + 20)
                section = '\n'.join(lines[start:end])
                fused_sections.append(f"--- Line {i} in {os.path.basename(filepath)} ---\n{section}\n")

        if fused_sections:
            out.write(f"\n{'='*60}\n")
            out.write(f"FILE: {os.path.basename(filepath)} ({size/1024/1024:.0f} MB)\n")
            out.write(f"Found {len(fused_sections)} fused_moe references\n")
            out.write(f"{'='*60}\n\n")
            # Write first 5 sections
            for s in fused_sections[:5]:
                out.write(s + '\n')
            print(f"  Found {len(fused_sections)} fused_moe references")
        else:
            print(f"  No fused_moe found")

print(f"\nExtracted to {out_file} ({os.path.getsize(out_file)/1024:.0f} KB)")

# Upload the extract to GCS
try:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket('vllm-cb-storage2')
    blob = bucket.blob('cuiq/mosaic_dump/fused_moe_hlo_extract.txt')
    blob.upload_from_filename(out_file)
    print(f"Uploaded to GCS: cuiq/mosaic_dump/fused_moe_hlo_extract.txt")
except Exception as e:
    print(f"GCS upload failed: {e}")

# Also print the extract to stdout (for BK logs)
with open(out_file) as f:
    print(f.read()[:5000])
PYEOF
