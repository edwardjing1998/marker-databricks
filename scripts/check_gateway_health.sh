#!/usr/bin/env bash
# Authenticated private health check. It never submits a Databricks job.
set -euo pipefail
LOG=$(mktemp)
PF_PID=''
cleanup() {
  if [[ -n "$PF_PID" ]]; then
    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
  fi
  rm -f "$LOG"
}
trap cleanup EXIT
oc port-forward --address=127.0.0.1 deployment/marker-databricks 18080:8080 > "$LOG" 2>&1 &
PF_PID=$!
for attempt in $(seq 1 30); do
  if ! kill -0 "$PF_PID" 2>/dev/null; then
    echo 'Private health port-forward failed. Check pods/portforward RBAC and local port 18080.' >&2
    exit 1
  fi
  if grep -q '^Forwarding from 127.0.0.1:18080' "$LOG" && \
     curl --fail --silent --show-error --max-time 5 http://127.0.0.1:18080/health/ready | \
     python -c 'import json,sys; body=json.load(sys.stdin); sys.exit(0 if body.get("status")=="UP" and body.get("application")=="marker-databricks" else 1)' >/dev/null 2>&1; then
    echo 'Private gateway health check passed. No Databricks run was started.'
    exit 0
  fi
  sleep 2
done
echo 'Private gateway health check timed out.' >&2
exit 1
