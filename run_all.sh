#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/min/test/EfficientNav"
DETECTION_SCRIPT="$ROOT_DIR/run_detection.sh"
PLANNER_SCRIPT="$ROOT_DIR/run_planner.sh"

"$DETECTION_SCRIPT" &
DETECTION_PID=$!

cleanup() {
  if kill -0 "$DETECTION_PID" >/dev/null 2>&1; then
    kill "$DETECTION_PID" >/dev/null 2>&1 || true
    wait "$DETECTION_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

sleep 2
"$PLANNER_SCRIPT"
