#!/bin/bash

BASE_MODEL="state-spaces/mamba-130m-hf"
CONFIG_PATH="model/eval_config.json"
EVAL_FILE="eval_subset.jsonl"
CHUNK_SIZE=4
BATCH_SIZE=300
OUTPUT_DIR="model/model3072_32_e6/evaluate"

echo "Evaluating base model"
python evaluate.py --base_model $BASE_MODEL  --config $CONFIG_PATH --eval_file $EVAL_FILE --chunk_size $CHUNK_SIZE --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR 

for step in $(seq 900 300 3300)
do
    echo "Evaluating step $step"
    CKPT_PATH="model/model3072_32_e6/step_$step"
    python evaluate.py --base_model $BASE_MODEL  --config $CONFIG_PATH --eval_file $EVAL_FILE --chunk_size $CHUNK_SIZE --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR --ckpt_path $CKPT_PATH 2>&1 | tee -a "${OUTPUT_DIR}/evaluation_output.log"
done

python evaluate.py --base_model $BASE_MODEL  --config $CONFIG_PATH --eval_file $EVAL_FILE --chunk_size $CHUNK_SIZE --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR 

echo "Evaluating final checkpoint"
CKPT_PATH="model/model3072_32_e6/step_3438"
python evaluate.py --base_model $BASE_MODEL  --config $CONFIG_PATH --eval_file $EVAL_FILE --chunk_size $CHUNK_SIZE --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR -ckpt_path $CKPT_PATH