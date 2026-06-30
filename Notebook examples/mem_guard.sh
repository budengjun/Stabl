#!/usr/bin/env bash
set -u

LIMIT_GB="${LIMIT_GB:-18}"
INTERVAL_SEC="${INTERVAL_SEC:-5}"
GRACE_SEC="${GRACE_SEC:-20}"

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  LIMIT_GB=18 INTERVAL_SEC=5 ./mem_guard.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -u run_cv_DREAM_fast.py"
  exit 2
fi

LIMIT_KB=$(( LIMIT_GB * 1024 * 1024 ))
PID=""
PGID=""

cleanup() {
  echo
  echo "===== Memory guard received stop signal ====="

  if [ -n "${PGID:-}" ]; then
    echo "Sending TERM to process group $PGID"
    kill -TERM "-$PGID" 2>/dev/null || true

    sleep 5

    if ps -eo pgid= | awk -v pgid="$PGID" '$1 == pgid {found=1} END {exit !found}'; then
      echo "Process group still alive. Sending KILL."
      kill -KILL "-$PGID" 2>/dev/null || true
    fi
  fi

  exit 130
}

trap cleanup INT TERM HUP

echo "===== Memory guard started ====="
echo "Limit: ${LIMIT_GB} GB"
echo "Interval: ${INTERVAL_SEC} sec"
echo "Grace period: ${GRACE_SEC} sec"
echo "Command: $*"
echo

setsid "$@" &
PID=$!

sleep 1

PGID="$(ps -o pgid= -p "$PID" | tr -d ' ')"

if [ -z "$PGID" ]; then
  echo "Failed to get process group id."
  exit 1
fi

echo "Started PID: $PID"
echo "Process group: $PGID"
echo

while kill -0 "$PID" 2>/dev/null; do
  RSS_KB="$(
    ps -eo pid=,pgid=,rss= |
    awk -v pgid="$PGID" '$2 == pgid {s += $3} END {print s + 0}'
  )"

  RSS_GB="$(awk "BEGIN {printf \"%.2f\", ${RSS_KB}/1024/1024}")"

  echo "$(date '+%Y-%m-%d %H:%M:%S')  RSS=${RSS_GB} GB"

  if [ "$RSS_KB" -gt "$LIMIT_KB" ]; then
    echo
    echo "===== MEMORY LIMIT EXCEEDED ====="
    echo "RSS=${RSS_GB} GB, limit=${LIMIT_GB} GB"
    echo "Sending TERM to process group $PGID"

    kill -TERM "-$PGID" 2>/dev/null || true

    sleep "$GRACE_SEC"

    if ps -eo pgid= | awk -v pgid="$PGID" '$1 == pgid {found=1} END {exit !found}'; then
      echo "Process group still alive. Sending KILL."
      kill -KILL "-$PGID" 2>/dev/null || true
    fi

    echo "Stopped because memory exceeded threshold."
    exit 137
  fi

  sleep "$INTERVAL_SEC"
done

wait "$PID"
EXIT_CODE=$?

trap - INT TERM HUP

echo
echo "===== Command finished ====="
echo "Exit code: $EXIT_CODE"

exit "$EXIT_CODE"
