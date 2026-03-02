#!/bin/bash
# Run from the root of your terraform directory.
# Creates alertmanager.zip ready for Terraform to upload.

set -euo pipefail

FUNC_DIR="./functions/alertmanager"
OUT="./alertmanager.zip"

echo "→ Zipping $FUNC_DIR → $OUT"
cd "$FUNC_DIR"
zip -r "../../alertmanager.zip" . -x "*.pyc" -x "__pycache__/*"
cd - > /dev/null

echo "✓ Created $OUT ($(du -sh $OUT | cut -f1))"
echo ""
echo "Now run:"
echo "  terraform plan"
echo "  terraform apply"
