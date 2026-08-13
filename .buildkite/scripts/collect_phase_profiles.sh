#!/bin/bash
# Collect phased-profiler traces from every host of a multi-host run and push
# them to GCS.
#
# Why this is needed: PhasedBasedProfiler writes with plain os.makedirs/open,
# so PHASED_PROFILING_DIR must be a LOCAL path -- a gs:// value silently
# creates a directory literally named "gs:" under the process CWD. Each host
# therefore keeps its own traces inside its own container. Pointing the
# profiler at the bind-mounted HF-home (container /root/.cache/huggingface ->
# host $HOST_HF_HOME) puts them on the host filesystem instead, where they
# survive `docker rm -f node`, and this script then uploads them per host.
#
# Usage: collect_phase_profiles.sh <gcs_dst> [host_profile_dir]
#   gcs_dst           e.g. gs://bucket/path/run1
#   host_profile_dir  default ${HOST_HF_HOME:-/tmp/hf_home}/phase_prof
set -uo pipefail

GCS_DST="${1:?usage: collect_phase_profiles.sh <gcs_dst> [host_profile_dir]}"
HOST_HF_HOME="${HOST_HF_HOME:-/tmp/hf_home}"
PROF_DIR="${2:-${HOST_HF_HOME}/phase_prof}"
SSH_USER="${SSH_USER:-$(whoami)}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null
          -o IPQoS=none -i /var/lib/buildkite-agent/.ssh/id_rsa)

# Reuse run_multihost.sh's discovery so head/worker ordering matches the run.
ALL_IPS=""
if command -v gcloud &>/dev/null; then
  ZONE="${ZONE:-$(curl -s -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/zone' | awk -F/ '{print $NF}')}"
  TPU_NAME="${TPU_NAME:-$(curl -s -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/description' 2>/dev/null || echo '')}"
  if [[ -n "$TPU_NAME" && -n "$ZONE" ]]; then
    ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
      --format="value(networkEndpoints[].ipAddress)" 2>/dev/null)
    ALL_IPS="${ALL_IPS//;/ }"; ALL_IPS="${ALL_IPS//,/ }"
  fi
fi
# shellcheck disable=SC2206
IPS=($ALL_IPS)
if [[ ${#IPS[@]} -eq 0 ]]; then
  echo "[collect-prof] WARNING: no IPs discovered; uploading local host only"
fi

upload_local() {  # label
  local label="$1"
  if [[ ! -d "$PROF_DIR" ]]; then
    echo "[collect-prof] $label: no $PROF_DIR (profiler may not have fired)"
    return 0
  fi
  local n
  n=$(find "$PROF_DIR" -name '*.xplane.pb' 2>/dev/null | wc -l)
  echo "[collect-prof] $label: $n xplane.pb, $(du -sh "$PROF_DIR" 2>/dev/null | cut -f1)"
  find "$PROF_DIR" -maxdepth 1 -mindepth 1 -type d -printf '  phase: %f\n' 2>/dev/null
  gsutil -m -q cp -r "$PROF_DIR"/* "${GCS_DST}/${label}/" \
    && echo "[collect-prof] $label: uploaded to ${GCS_DST}/${label}/" \
    || echo "[collect-prof] $label: UPLOAD FAILED"
}

echo "[collect-prof] destination: ${GCS_DST}"
echo "[collect-prof] host profile dir: ${PROF_DIR}"

# Head is this machine (the buildkite agent runs on it).
upload_local "head"

# Workers: everything after the first discovered IP.
for ip in "${IPS[@]:1}"; do
  echo "[collect-prof] --- worker ${ip} ---"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
    "PROF_DIR='${PROF_DIR}' GCS_DST='${GCS_DST}' bash -s" <<'REMOTE' || echo "[collect-prof] worker ${ip}: ssh/upload failed"
    if [ ! -d "$PROF_DIR" ]; then echo "  no $PROF_DIR on this worker"; exit 0; fi
    n=$(find "$PROF_DIR" -name '*.xplane.pb' 2>/dev/null | wc -l)
    echo "  $n xplane.pb, $(du -sh "$PROF_DIR" 2>/dev/null | cut -f1)"
    gsutil -m -q cp -r "$PROF_DIR"/* "${GCS_DST}/worker_$(hostname)/" \
      && echo "  uploaded to ${GCS_DST}/worker_$(hostname)/" || echo "  UPLOAD FAILED"
REMOTE
done

echo "[collect-prof] final listing:"
gsutil ls -r "${GCS_DST}/**/*.xplane.pb" 2>/dev/null | head -40 || echo "  (nothing uploaded)"
