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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
GCS_BUCKET="${GCS_BUCKET:-gs://vllm-cb-storage2/cuiq/sweep}"
ENABLE_EP="${ENABLE_EXPERT_PARALLEL:-0}"
# USE_MOE_EP_KERNEL is read directly as env var by tpu_inference

# Optional server args
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

# Optional client args
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-}"
REQUEST_RATE="${REQUEST_RATE:-}"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# Determine track name
if [ "${USE_MOE_EP_KERNEL:-0}" = "1" ] && [ "$ENABLE_EP" = "1" ]; then
  TRACK="fused_ep"
elif [ "$ENABLE_EP" = "1" ]; then
  TRACK="gmm_ep"
else
  TRACK="tp"
fi
TAG="${MODEL_SHORT}_${TRACK}_bt${BT}_s${S}"
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
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN ISL=$ISL OSL=$OSL (ISL+OSL=$((ISL+OSL)) must be <= $MAX_MODEL_LEN)"
echo "  TRACK=$TRACK ENABLE_EP=$ENABLE_EP USE_MOE_EP_KERNEL=${USE_MOE_EP_KERNEL:-0}"
if [ $((ISL + OSL)) -gt "$MAX_MODEL_LEN" ]; then
  echo "  FATAL: ISL+OSL=$((ISL+OSL)) exceeds max-model-len=$MAX_MODEL_LEN"
  exit 1
fi

echo "=== Validation passed ==="

# Build optional server flags
EP_FLAG=""
if [ "$ENABLE_EP" = "1" ]; then
  EP_FLAG="--enable-expert-parallel"
fi
OPT_SERVER_FLAGS=""
[ -n "$KV_CACHE_DTYPE" ] && OPT_SERVER_FLAGS="$OPT_SERVER_FLAGS --kv-cache-dtype=$KV_CACHE_DTYPE"
[ -n "$GPU_MEM_UTIL" ] && OPT_SERVER_FLAGS="$OPT_SERVER_FLAGS --gpu-memory-utilization=$GPU_MEM_UTIL"
OPT_SERVER_FLAGS="$OPT_SERVER_FLAGS $EXTRA_SERVER_ARGS"

# Build profiler args
PROFILER_ARGS=""
PROFILER_ENV=""
if [ -n "${PHASED_PROFILING_DIR:-}" ]; then
  # Use phased profiling (auto-captures decode/prefill phases, gets device traces)
  PROFILER_ENV="PHASED_PROFILING_DIR=$WORKDIR/phased_profile"
  mkdir -p "$WORKDIR/phased_profile"
  echo "  Using PHASED_PROFILING_DIR for device-level profiling"
else
  # Use --profiler-config (client-triggered)
  PROFILER_ARGS="--profiler-config {\"profiler\": \"torch\", \"torch_profiler_dir\": \"$WORKDIR/profile\"}"
fi

# Start server in background
SERVER_CMD="MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=${USE_MOE_EP_KERNEL:-0} $PROFILER_ENV vllm serve $MODEL \
  --download-dir=/tmp/hf_home \
  --tensor_parallel_size=$TP \
  --max-model-len=$MAX_MODEL_LEN \
  --max-num-batched-tokens=$BT \
  --max-num-seqs=$S \
  $EP_FLAG \
  $PROFILER_ARGS"
echo "=== Server command ==="
echo "  $SERVER_CMD"
echo ""

# Export profiling env vars for the server process
export MODEL_IMPL_TYPE=vllm
export USE_MOE_EP_KERNEL=${USE_MOE_EP_KERNEL:-0}
if [ -n "${PHASED_PROFILING_DIR:-}" ]; then
  export PHASED_PROFILING_DIR="$WORKDIR/phased_profile"
fi

# Build the full server command as an array to preserve quoting
SERVER_ARGS=(
  --download-dir=/tmp/hf_home
  --tensor_parallel_size=$TP
  --max-model-len=$MAX_MODEL_LEN
  --max-num-batched-tokens=$BT
  --max-num-seqs=$S
)
[ -n "$EP_FLAG" ] && SERVER_ARGS+=($EP_FLAG)
[ -n "$KV_CACHE_DTYPE" ] && SERVER_ARGS+=(--kv-cache-dtype=$KV_CACHE_DTYPE)
[ -n "$GPU_MEM_UTIL" ] && SERVER_ARGS+=(--gpu-memory-utilization=$GPU_MEM_UTIL)
# Add extra server args (split by spaces)
for arg in $EXTRA_SERVER_ARGS; do
  SERVER_ARGS+=("$arg")
done
# Add profiler config (properly quoted JSON)
if [ -z "${PHASED_PROFILING_DIR:-}" ]; then
  SERVER_ARGS+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$WORKDIR/profile\"}")
fi

echo "=== Server args ==="
echo "  ${SERVER_ARGS[*]}"

vllm serve "$MODEL" "${SERVER_ARGS[@]}" > "$WORKDIR/server.log" 2>&1 &
SERVER_PID=$!

echo "  Server PID=$SERVER_PID, waiting for ready..."

# Wait for server (up to 45 min for large models like 480B which compile many sizes ~85s each)
READY=0
for i in $(seq 1 540); do
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
  echo "  ERROR: Server not ready after 45min"
  tail -20 "$WORKDIR/server.log"
  kill -9 $SERVER_PID 2>/dev/null || true
  exit 1
fi

# Run benchmark
PROFILE_FLAG=""
if [ -z "${PHASED_PROFILING_DIR:-}" ]; then
  PROFILE_FLAG="--profile"
fi
OPT_CLIENT_FLAGS=""
[ -n "$MAX_CONCURRENCY" ] && OPT_CLIENT_FLAGS="$OPT_CLIENT_FLAGS --max-concurrency=$MAX_CONCURRENCY"
[ -n "$RANDOM_RANGE_RATIO" ] && OPT_CLIENT_FLAGS="$OPT_CLIENT_FLAGS --random-range-ratio=$RANDOM_RANGE_RATIO"
[ -n "$REQUEST_RATE" ] && OPT_CLIENT_FLAGS="$OPT_CLIENT_FLAGS --request-rate=$REQUEST_RATE"
CLIENT_CMD="vllm bench serve --backend vllm --model $MODEL \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL \
  --num-prompts $NUM_PROMPTS $PROFILE_FLAG $OPT_CLIENT_FLAGS"
echo "=== Client command ==="
echo "  $CLIENT_CMD"
echo ""

vllm bench serve --backend vllm --model "$MODEL" \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL \
  --num-prompts $NUM_PROMPTS $PROFILE_FLAG $OPT_CLIENT_FLAGS \
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

# Find traces from either --profiler-config or PHASED_PROFILING_DIR
TRACE=$(find "$WORKDIR" -name "*.trace.json.gz" -path "*/plugins/profile/*" 2>/dev/null | head -1)
XPLANE=$(find "$WORKDIR" -name "*.xplane.pb" -path "*/plugins/profile/*" 2>/dev/null | head -1)

# Try python3 google-cloud-storage (most likely available in the image)
python3 -c "
from google.cloud import storage
import sys, os

bucket_name = 'vllm-cb-storage2'
prefix = 'cuiq/sweep/${TAG}'
client = storage.Client()
bucket = client.bucket(bucket_name)

files = {
    'server.log': '${WORKDIR}/server.log',
    'client.log': '${WORKDIR}/client.log',
}
trace = '${TRACE}'
xplane = '${XPLANE}'
if trace:
    files['trace.json.gz'] = trace
if xplane:
    files['xplane.pb'] = xplane

for name, path in files.items():
    if os.path.exists(path):
        blob = bucket.blob(f'{prefix}/{name}')
        blob.upload_from_filename(path)
        print(f'  Uploaded {name} ({os.path.getsize(path)/1e6:.1f} MB)')
    else:
        print(f'  SKIP {name} (not found)')
print('  GCS upload complete')
" 2>&1 || echo "  WARNING: python GCS upload failed, trying gsutil/gcloud..."

# Fallback to gsutil/gcloud
if ! python3 -c "from google.cloud import storage" 2>/dev/null; then
  if command -v gsutil &>/dev/null; then
    gsutil -m cp "$WORKDIR/server.log" "${GCS_PATH}/server.log" 2>/dev/null
    gsutil -m cp "$WORKDIR/client.log" "${GCS_PATH}/client.log" 2>/dev/null
    [ -n "$TRACE" ] && gsutil -m cp "$TRACE" "${GCS_PATH}/trace.json.gz" 2>/dev/null
    [ -n "$XPLANE" ] && gsutil -m cp "$XPLANE" "${GCS_PATH}/xplane.pb" 2>/dev/null
  elif command -v gcloud &>/dev/null; then
    gcloud storage cp "$WORKDIR/server.log" "${GCS_PATH}/server.log" 2>/dev/null
    gcloud storage cp "$WORKDIR/client.log" "${GCS_PATH}/client.log" 2>/dev/null
    [ -n "$TRACE" ] && gcloud storage cp "$TRACE" "${GCS_PATH}/trace.json.gz" 2>/dev/null
    [ -n "$XPLANE" ] && gcloud storage cp "$XPLANE" "${GCS_PATH}/xplane.pb" 2>/dev/null
  else
    echo "  WARNING: No GCS upload method available"
  fi
fi

# Print parseable summary line (can be extracted from Buildkite logs)
REQ_TPUT=$(grep "Request throughput" "$WORKDIR/client.log" | grep -oP '[\d.]+' | head -1)
OUT_TPUT=$(grep "Output token throughput" "$WORKDIR/client.log" | grep -oP '[\d.]+' | head -1)
TTFT=$(grep "Mean TTFT" "$WORKDIR/client.log" | grep -oP '[\d.]+' | head -1)
TPOT=$(grep "Mean TPOT" "$WORKDIR/client.log" | grep -oP '[\d.]+' | head -1)
KV_AVG=$(grep "KV cache usage" "$WORKDIR/server.log" | grep -oP '[\d.]+(?=%)' | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
RUNNING_AVG=$(grep "Running:" "$WORKDIR/server.log" | grep -oP 'Running: \K[\d]+' | awk '{s+=$1; n++} END {if(n>0) printf "%.0f", s/n; else print "N/A"}')

echo ""
echo "SWEEP_RESULT|${TAG}|${BT}|${S}|${REQ_TPUT}|${OUT_TPUT}|${TTFT}|${TPOT}|${KV_AVG}|${RUNNING_AVG}"
echo ""
echo "=== Done: $TAG ==="
