#!/usr/bin/env bash
# Convenience wrapper for the local test pipeline.
# Run from anywhere — paths are resolved relative to this script.
#
# Usage:
#   ./run_local.sh                        # webcam 0, cam1, 1 fps
#   ./run_local.sh --source 0             # explicit webcam
#   ./run_local.sh --source video.mp4
#   ./run_local.sh --source /path/to/frames/
#   ./run_local.sh --source frame.jpg
#   ./run_local.sh --source 0 --cameras 3  # 3 webcams in parallel
#   ./run_local.sh --fps 2.0 --conf 0.4
#
# Optional env overrides (export before running or inline):
#   MODEL_PATH=../terraform_v2/services/judge/best.pt
#   GMAIL_ADDRESS=you@gmail.com
#   GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

set -euo pipefail
cd "$(dirname "$0")"

export MODEL_PATH="${MODEL_PATH:-../terraform_v2/services/judge/best.pt}"
export CONF_THRESHOLD="${CONF_THRESHOLD:-0.35}"
export COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-60}"

python3 orchestrator.py "$@"
