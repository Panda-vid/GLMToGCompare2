#!/bin/bash
export PYTHONPATH=.

SCRIPT_DIR="$(dirname "$0")"
# Define input variables for the Python program
ENCODER_MODELCARD="plenz/GLM-flan-t5-base"
GENERATOR_MODELCARD="google/flan-t5-base"
TRAIN_FILE="./data/preprocessed/trex-train-kilt.jsonl"
SAVE_LOCATION="./saved_models/test/flan-t5-base"
PROBLEM_TYPE="classification"
GLM_TYPE="global"
DEVICE="cuda"
BATCH_SIZE=16
OPTIMIZER="AdamW"
LEARNING_RATE="1e-4"
NUM_EPOCHS=5
EARLY_STOPPING=2
NEIGHBORHOOD_SIZE=10
EVAL_FILE="./data/preprocessed/trex-eval-kilt.jsonl"
CHECKPOINTING_INTERVAL=500

# Path to the Python program
PYTHON_PROGRAM="./GraphLanguageModel/train_glm.py"

# Start the Python program with inputs
echo starting python3 "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"
accelerate launch "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -c "$CHECKPOINTING_INTERVAL"