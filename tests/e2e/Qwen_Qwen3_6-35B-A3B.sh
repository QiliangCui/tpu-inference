#!/bin/bash
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

set -ex

TEST_MODEL="${TEST_MODEL:-Qwen/Qwen3.6-35B-A3B}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MINIMUM_ACCURACY_THRESHOLD="${MINIMUM_ACCURACY_THRESHOLD:-0.90}"
USE_BATCHED_RPA_KERNEL="${USE_BATCHED_RPA_KERNEL:-1}"

# Qwen3.6 is a hybrid-thinking reasoning model: GSM8K needs the chat
# template, an 8192 context for multiturn fewshot prompts plus thinking
# tokens, a 4096-token generation budget, and an <|im_end|> stop override
# (the stock "Question:" stop list truncates thinking-mode generations).
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-8192}"
EVAL_GEN_KWARGS="${EVAL_GEN_KWARGS:-max_gen_toks=4096,until=<|im_end|>}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"

export TEST_MODEL TENSOR_PARALLEL_SIZE MINIMUM_ACCURACY_THRESHOLD \
  USE_BATCHED_RPA_KERNEL USE_CHAT_TEMPLATE EVAL_MAX_MODEL_LEN \
  EVAL_GEN_KWARGS EVAL_LIMIT

# Unit test: verify the model loads and runs offline inference
SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  python /workspace/tpu_inference/examples/offline_inference.py \
    --model "${TEST_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-num-batched-tokens 4096 \
    --max-model-len 4096

# Accuracy test
bash /workspace/tpu_inference/tests/e2e/benchmarking/test_accuracy.sh
