#!/bin/bash
# POC experiments
# Run with: bash scripts/run_poc.sh

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(dirname $(dirname $(realpath $0)))/../../"

python -m poc.run --model both --epochs 30 --seeds 42 123 456
