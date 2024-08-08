#!/bin/bash

BASE_MODEL=$1
EVAL_FILE=$2
CHUNK_SIZE=$3
BATCH_SIZE=$4
MODEL_DIR=$5
OUTPUT_DIR="${MODEL_DIR}/evaluate"

mkdir -p "$OUTPUT_DIR"

echo "Evaluating base model"
python evaluate.py --base_model $BASE_MODEL --eval_file $EVAL_FILE --chunk_size $CHUNK_SIZE --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR 

for dir in "$MODEL_DIR"/*/; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")        
        if [[ $dir_name == step_* ]]; then
            echo "Evaluating $dir_name"
            CKPT_PATH="model/model3072_32_e6/step_$step"
            python evaluate.py --base_model $BASE_MODEL --eval_file $EVAL_FILE --chunk_size $CHUNK_SIZE --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR --ckpt_path $dir_name 2>&1 | tee -a "${OUTPUT_DIR}/evaluation_output.log"
        fi
    fi
done