#!/bin/bash

MODEL_PATH_OR_NAME=sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco
MODEL_OUT=/projects/0/prjs0907/multivariate_ir/experiments/tas_b_zeroshot
EXTRA_ARGS=""
LOG_FILE=/projects/0/prjs0907/multivariate_ir/experiments/tas_b_zeroshot/eval.log


export MODEL_PATH_OR_NAME=${MODEL_PATH_OR_NAME}
export MODEL_OUT=${MODEL_OUT}
export EXTRA_ARGS=${EXTRA_ARGS}
export LOG_FILE=${LOG_FILE}


/bin/bash eval_all.sh
