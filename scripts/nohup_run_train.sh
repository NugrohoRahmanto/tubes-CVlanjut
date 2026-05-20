#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
uname -a
#date
#env
date

mkdir -p logs

# Override GPU_IDS/NPROC_PER_NODE before running if needed, for example:
# GPU_IDS=0 NPROC_PER_NODE=1 sh scripts/nohup_run_train.sh
nohup sh run_train_demo.sh > ./logs/log_run_train_demo 2>&1 &
