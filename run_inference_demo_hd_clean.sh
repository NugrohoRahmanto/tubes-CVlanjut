#!/bin/bash
set -eu

cd "$(dirname "$0")"
uname -a
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

# HD visualization demo settings.
STAGE=${STAGE:-inference}
MODEL=${MODEL:-./pretrained_model/docsam_large_all_dataset.pth}
MODEL_SIZE=${MODEL_SIZE:-large}
SAVE_PATH=${SAVE_PATH:-./outputs/outputs_inference_demo_hd_clean/}
MAX_NUM=${MAX_NUM:-10}

SHORT_RANGE=${SHORT_RANGE:-704,896}
PATCH_SIZE=${PATCH_SIZE:-640,640}
PATCH_NUM=${PATCH_NUM:-1}
KEEP_SIZE=${KEEP_SIZE:-False}
PREDICT_BATCH_SIZE=${PREDICT_BATCH_SIZE:-2}
VISUAL_SCORE_THRESHOLD=${VISUAL_SCORE_THRESHOLD:-0.5}
VISUAL_MASK_ALPHA=${VISUAL_MASK_ALPHA:-0.28}
VISUAL_USE_ORIGINAL=${VISUAL_USE_ORIGINAL:-True}
VISUAL_BBOX_ONLY=${VISUAL_BBOX_ONLY:-True}

GPU_IDS=${GPU_IDS:-0}
DATASET=${DATASET:-WTW}

case "${DATASET}" in
    None|none|NONE)
        set -- "${EVAL_PATH_1}" "${EVAL_PATH_2}" "${EVAL_PATH_3}" "${EVAL_PATH_4}" "${EVAL_PATH_5}" "${EVAL_PATH_6}"
        ;;
    PubLayNet|publaynet|layout|Layout)
        set -- "${EVAL_PATH_1}"
        ;;
    Style1|style1|CASIA-AHCDB-Style1)
        set -- "${EVAL_PATH_2}"
        ;;
    Style2|style2|CASIA-AHCDB-Style2)
        set -- "${EVAL_PATH_3}"
        ;;
    CASIA-HWDB|casia-hwdb|hwdb|HWDB)
        set -- "${EVAL_PATH_4}"
        ;;
    WTW|wtw)
        set -- "${EVAL_PATH_5}"
        ;;
    TotalText|totaltext|Total-Text)
        set -- "${EVAL_PATH_6}"
        ;;
    *)
        echo "Unknown DATASET: ${DATASET}. Use WTW, PubLayNet, Style1, Style2, CASIA-HWDB, TotalText, or None." >&2
        exit 1
        ;;
esac

for path in "$@"; do
    [ -d "${path}" ] || { echo "Missing dataset folder: ${path}" >&2; exit 1; }
done
[ -f "${MODEL}" ] || { echo "Missing checkpoint: ${MODEL}" >&2; exit 1; }
mkdir -p "${SAVE_PATH}" logs outputs temp

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
CUDA_VISIBLE_DEVICES=${GPU_IDS} "${PYTHON:-python}" -B -u test.py \
    --eval-path "$@" \
    --stage ${STAGE} --restore-from ${MODEL} --model-size ${MODEL_SIZE} --save-path ${SAVE_PATH} --max-num ${MAX_NUM} \
    --short-range ${SHORT_RANGE} --patch-size ${PATCH_SIZE} --patch-num ${PATCH_NUM} --keep-size ${KEEP_SIZE} \
    --predict-batch-size ${PREDICT_BATCH_SIZE} \
    --visual-score-threshold ${VISUAL_SCORE_THRESHOLD} --visual-mask-alpha ${VISUAL_MASK_ALPHA} --visual-use-original ${VISUAL_USE_ORIGINAL} \
    --visual-bbox-only ${VISUAL_BBOX_ONLY} \
    --gpus ${GPU_IDS}
