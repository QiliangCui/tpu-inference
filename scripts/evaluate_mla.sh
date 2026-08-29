#!/bin/bash
# Evaluate a bench_mla.sh log: extract the (last) ==RESULT START== /
# ==RESULT END== JSON block and print the number of cases, the average
# latency across successful cases, and a reward score:
#     reward = 1000 - average_latency_us     (lower latency = higher reward)
#
# Usage:
#   scripts/evaluate_mla.sh bench.log            # from a saved log file
#   scripts/bench_mla.sh | scripts/evaluate_mla.sh   # piped directly
#
# Example output:
#   cases: 37
#   average latency: 717.64 us
#   reward: 282.36
#
# Failed cases (entries with "error") are counted but excluded from the
# average. Exits non-zero if the input has no result block or no successful
# case. Note: the average is only comparable between runs of the SAME case
# set — don't compare averages taken with different --filter values.
set -euo pipefail

if [ $# -ge 1 ]; then
    input="$1"
else
    # The heredoc below occupies stdin, so buffer piped input to a temp file.
    input=$(mktemp)
    trap 'rm -f "$input"' EXIT
    cat > "$input"
fi

python3 - "$input" << 'EOF'
import json
import sys

text = open(sys.argv[1]).read()

start_marker, end_marker = "==RESULT START==", "==RESULT END=="
start = text.rfind(start_marker)
end = text.rfind(end_marker)
if start == -1 or end == -1 or end < start:
    sys.exit("error: no ==RESULT START==/==RESULT END== block found in input")

results = json.loads(text[start + len(start_marker):end])
latencies = [r["latency_us"] for r in results if "latency_us" in r]
failed = [r for r in results if "latency_us" not in r]

if not latencies:
    sys.exit(f"error: no successful cases in result ({len(failed)} failed)")

print(f"cases: {len(latencies)}" +
      (f" (+{len(failed)} failed, excluded)" if failed else ""))
avg = sum(latencies) / len(latencies)
print(f"average latency: {avg:.2f} us")
print(f"reward: {1000.0 - avg:.2f}")
EOF
