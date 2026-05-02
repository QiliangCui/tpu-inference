#!/bin/bash
# Sweep profiling: run one benchmark with profiling, upload results to GCS.
# Called from Buildkite pipeline with env vars:
#   TEST_MODEL, TENSOR_PARALLEL_SIZE, MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS
# Uploads to gs://vllm-cb-storage2/cuiq/sweep/{tag}/

set -euo pipefail

MODEL="${TEST_MODEL:?TEST_MODEL required}"
TP="${TENSOR_PARALLEL_SIZE:-2}"
BT="${MAX_NUM_BATCHED_TOKENS:?MAX_NUM_BATCHED_TOKENS required}"
S="${MAX_NUM_SEQS:?MAX_NUM_SEQS required}"
ISL="${INPUT_LEN:-1800}"
OSL="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
GCS_BUCKET="${GCS_BUCKET:-gs://vllm-cb-storage2/cuiq/sweep}"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
TAG="${MODEL_SHORT}_bt${BT}_s${S}"
WORKDIR="/tmp/sweep_${TAG}"
mkdir -p "$WORKDIR"

echo "=== Sweep Run: $TAG ==="
echo "  Model=$MODEL TP=$TP BT=$BT S=$S ISL=$ISL OSL=$OSL prompts=$NUM_PROMPTS"

# --- Validation ---
echo "=== Validation ==="

# Check TPU devices match TP
NUM_DEVICES=$(python3 -c "import jax; print(jax.device_count())" 2>/dev/null || echo "0")
echo "  JAX devices: $NUM_DEVICES, TP requested: $TP"
if [ "$NUM_DEVICES" -lt "$TP" ]; then
  echo "  FATAL: Only $NUM_DEVICES devices but TP=$TP requested"
  exit 1
fi

# Check vllm and tpu_inference are importable
VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "MISSING")
TPI_VER=$(python3 -c "import tpu_inference; print(tpu_inference.__version__)" 2>/dev/null || echo "MISSING")
echo "  vllm=$VLLM_VER tpu_inference=$TPI_VER"

# Check model is downloadable (just tokenizer, fast check)
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL', cache_dir='/tmp/hf_home', trust_remote_code=True)
print(f'  Tokenizer loaded: vocab_size={tok.vocab_size}')
" 2>/dev/null || echo "  WARNING: tokenizer pre-check failed (may still work via vllm)"

# Check GCS is writable
TEST_GCS="${GCS_BUCKET}/_validation_test"
if gsutil cp /dev/null "${TEST_GCS}" 2>/dev/null; then
  gsutil rm "${TEST_GCS}" 2>/dev/null
  echo "  GCS writable: OK"
elif gcloud storage cp /dev/null "${TEST_GCS}" 2>/dev/null; then
  gcloud storage rm "${TEST_GCS}" 2>/dev/null
  echo "  GCS writable: OK (gcloud)"
else
  echo "  WARNING: GCS write test failed — results may not upload"
fi

# Print all env vars for debugging
echo "  MAX_MODEL_LEN=2048 ISL=$ISL OSL=$OSL (ISL+OSL=$((ISL+OSL)) must be <= 2048)"
if [ $((ISL + OSL)) -gt 2048 ]; then
  echo "  FATAL: ISL+OSL=$((ISL+OSL)) exceeds max-model-len=2048"
  exit 1
fi

echo "=== Validation passed ==="

# Start server in background
MODEL_IMPL_TYPE=vllm vllm serve "$MODEL" \
  --download-dir=/tmp/hf_home \
  --tensor_parallel_size=$TP \
  --max-model-len=2048 \
  --max-num-batched-tokens=$BT \
  --max-num-seqs=$S \
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$WORKDIR/profile\"}" \
  > "$WORKDIR/server.log" 2>&1 &
SERVER_PID=$!

echo "  Server PID=$SERVER_PID, waiting for ready..."

# Wait for server (up to 20 min for large models)
READY=0
for i in $(seq 1 240); do
  if grep -q "Application startup complete" "$WORKDIR/server.log" 2>/dev/null; then
    READY=1
    echo "  Server ready after $((i*5))s"
    # Verify server picked up correct params
    echo "  --- Server config verification ---"
    grep "non-default args" "$WORKDIR/server.log" | head -1 | grep -oP "max_num_batched_tokens=\d+" || true
    grep "non-default args" "$WORKDIR/server.log" | head -1 | grep -oP "max_num_seqs=\d+" || true
    grep "non-default args" "$WORKDIR/server.log" | head -1 | grep -oP "tensor_parallel_size=\d+" || true
    grep "non-default args" "$WORKDIR/server.log" | head -1 | grep -oP "max_seq_len=\d+" || true
    grep "KV cache size" "$WORKDIR/server.log" | head -1 || true
    echo "  ---"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "  ERROR: Server died. Last 20 lines:"
    tail -20 "$WORKDIR/server.log"
    exit 1
  fi
  sleep 5
done

if [ $READY -eq 0 ]; then
  echo "  ERROR: Server not ready after 20min"
  tail -20 "$WORKDIR/server.log"
  kill -9 $SERVER_PID 2>/dev/null || true
  exit 1
fi

# Run benchmark
echo "  Running benchmark ($NUM_PROMPTS prompts, ISL=$ISL, OSL=$OSL)..."
vllm bench serve --backend vllm --model "$MODEL" \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL \
  --num-prompts $NUM_PROMPTS --profile \
  > "$WORKDIR/client.log" 2>&1
BENCH_EXIT=$?

# Kill server
kill -9 $SERVER_PID 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true

if [ $BENCH_EXIT -ne 0 ]; then
  echo "  ERROR: Benchmark failed (exit=$BENCH_EXIT). Last 10 lines:"
  tail -10 "$WORKDIR/client.log"
fi

# Print summary
echo "=== Results ==="
grep -E "Request throughput|Output token throughput|Mean TTFT|Mean TPOT" "$WORKDIR/client.log" || echo "  No throughput results"
echo ""
grep "KV cache usage" "$WORKDIR/server.log" | tail -5 || echo "  No KV cache stats"

# Upload to GCS
GCS_PATH="${GCS_BUCKET}/${TAG}"
echo ""
echo "=== Uploading to $GCS_PATH ==="

# Install gsutil if not available
if ! command -v gsutil &>/dev/null; then
  echo "  gsutil not found, trying gcloud..."
  if command -v gcloud &>/dev/null; then
    gcloud storage cp "$WORKDIR/server.log" "${GCS_PATH}/server.log"
    gcloud storage cp "$WORKDIR/client.log" "${GCS_PATH}/client.log"
    # Upload xprof profile (the plugins/profile one)
    TRACE=$(find "$WORKDIR/profile" -name "*.trace.json.gz" -path "*/plugins/profile/*" 2>/dev/null | head -1)
    XPLANE=$(find "$WORKDIR/profile" -name "*.xplane.pb" -path "*/plugins/profile/*" 2>/dev/null | head -1)
    [ -n "$TRACE" ] && gcloud storage cp "$TRACE" "${GCS_PATH}/trace.json.gz"
    [ -n "$XPLANE" ] && gcloud storage cp "$XPLANE" "${GCS_PATH}/xplane.pb"
  else
    echo "  WARNING: No gsutil or gcloud available, skipping upload"
  fi
else
  gsutil -m cp "$WORKDIR/server.log" "${GCS_PATH}/server.log"
  gsutil -m cp "$WORKDIR/client.log" "${GCS_PATH}/client.log"
  TRACE=$(find "$WORKDIR/profile" -name "*.trace.json.gz" -path "*/plugins/profile/*" 2>/dev/null | head -1)
  XPLANE=$(find "$WORKDIR/profile" -name "*.xplane.pb" -path "*/plugins/profile/*" 2>/dev/null | head -1)
  [ -n "$TRACE" ] && gsutil -m cp "$TRACE" "${GCS_PATH}/trace.json.gz"
  [ -n "$XPLANE" ] && gsutil -m cp "$XPLANE" "${GCS_PATH}/xplane.pb"
fi

echo "=== Done: $TAG ==="
