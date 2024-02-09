#!/bin/bash

MODEL_PATH_OR_NAME=$1
MODEL_OUT=$2
EXTRA_ARGS=$3
LOG_FILE=$4


if [ -z "$MODEL_PATH_OR_NAME" ] || [ -z "$MODEL_OUT" ] || [ -z "$LOG_FILE" ]
then
  echo "usage: MODEL_PATH_OR_NAME MODEL_OUT EXTRA_ARGS LOG_FILE"
  exit -1
fi

export MODEL_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_model_cache
export DATA_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_data_cache
export IR_DATASETS_HOME=/ivi/ilps/projects/multivariate_ir/.ird_cache/data
export IR_DATASETS_TMP=/ivi/ilps/projects/multivariate_ir/.ird_cache/temp
export HF_HOME=/ivi/ilps/projects/multivariate_ir/.hf_data_cache

export MODEL_PATH_OR_NAME=${MODEL_PATH_OR_NAME}
export MODEL_OUT=${MODEL_OUT}
export EXTRA_ARGS=${EXTRA_ARGS}
export LOG_FILE=${LOG_FILE}


/bin/bash eval_all.sh