#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="printermonitor-488112"
TOPIC="frames-in"
SUBSCRIPTION="frames-in-debug-local"

if gcloud pubsub subscriptions describe "$SUBSCRIPTION" \
     --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "Subscription already exists: $SUBSCRIPTION"
else
  gcloud pubsub subscriptions create "$SUBSCRIPTION" \
    --topic="$TOPIC" \
    --project="$PROJECT_ID" \
    --ack-deadline=60 \
    --message-retention-duration=10m
  echo "Created subscription: $SUBSCRIPTION"
fi

cat <<'EOF'

Next steps:

  cd debugging
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  gcloud auth application-default login

  # Run the local debug subscriber (uses repo's best.pt, GPU device cuda:0):
  python run.py --save-frames --show

  # Common flags:
  #   --conf 0.20             confidence threshold (matches v4 cloud env)
  #   --device cuda:0         torch device (cuda:0 / cpu)
  #   --model PATH            override path to best.pt
  #   --save-frames           dump raw + annotated frames to ./frames/
  #   --show                  live OpenCV window per camera

When done debugging, delete the subscription to stop accruing message backlog:
  gcloud pubsub subscriptions delete frames-in-debug-local --project=printermonitor-488112
EOF
