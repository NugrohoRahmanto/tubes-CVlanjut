#!/bin/bash
set -eu

cd "$(dirname "$0")"
uname -a
#date
#env
date

# Replace to your own data path
DATA_PATH=${DATA_PATH:-./data/demo_data}

# Layout
EVAL_PATH_1=${DATA_PATH}/Layout/PubLayNet/data/PubLayNet/test/

# Ancient
EVAL_PATH_2=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style1/test/
EVAL_PATH_3=${DATA_PATH}/Ancient/CASIA-AHCDB/data/Style2/test/

# Handwritten
EVAL_PATH_4=${DATA_PATH}/Handwritten/CASIA-HWDB/data/CASIA-HWDB/test/

# Table
EVAL_PATH_5=${DATA_PATH}/Table/WTW/data/WTW/test/

# SceneText
EVAL_PATH_6=${DATA_PATH}/SceneText/Total-Text/data/TotalText/test/


# Test settings. You can modify them to fit your own resources.
STAGE=${STAGE:-test}
MODEL=${MODEL:-./pretrained_model/docsam_large_all_dataset.pth}
MODEL_SIZE=${MODEL_SIZE:-large}
SAVE_PATH=${SAVE_PATH:-./outputs/outputs_test/demo/}
MAX_NUM=${MAX_NUM:-10}

SHORT_RANGE=${SHORT_RANGE:-704,896}
PATCH_SIZE=${PATCH_SIZE:-640,640}
PATCH_NUM=${PATCH_NUM:-1}
KEEP_SIZE=${KEEP_SIZE:-False}
VISUAL_SCORE_THRESHOLD=${VISUAL_SCORE_THRESHOLD:-0.5}
VISUAL_MASK_ALPHA=${VISUAL_MASK_ALPHA:-0.28}
VISUAL_USE_ORIGINAL=${VISUAL_USE_ORIGINAL:-True}

GPU_IDS=${GPU_IDS:-0}

set -- "${EVAL_PATH_1}" "${EVAL_PATH_2}" "${EVAL_PATH_3}" "${EVAL_PATH_4}" "${EVAL_PATH_5}" "${EVAL_PATH_6}"

for path in "$@"; do
    [ -d "${path}" ] || { echo "Missing dataset folder: ${path}" >&2; exit 1; }
done
[ -f "${MODEL}" ] || { echo "Missing checkpoint: ${MODEL}" >&2; exit 1; }
mkdir -p "${SAVE_PATH}" logs outputs temp

export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=${GPU_IDS} "${PYTHON:-python}" -B -u test.py \
    --eval-path "$@" \
    --stage ${STAGE} --restore-from ${MODEL} --model-size ${MODEL_SIZE} --save-path ${SAVE_PATH} --max-num ${MAX_NUM} \
    --short-range ${SHORT_RANGE} --patch-size ${PATCH_SIZE} --patch-num ${PATCH_NUM} --keep-size ${KEEP_SIZE} \
    --visual-score-threshold ${VISUAL_SCORE_THRESHOLD} --visual-mask-alpha ${VISUAL_MASK_ALPHA} --visual-use-original ${VISUAL_USE_ORIGINAL} \
    --gpus ${GPU_IDS}
