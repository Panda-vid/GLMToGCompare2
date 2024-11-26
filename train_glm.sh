#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --time=120
#SBATCH --mem=256gb
#SBATCH --job-name=trex-large
#SBATCH --gres=gpu:4
#SBATCH -o "$HOME/trex-large.o"
#SBATCH --error="$HOME/trex-large.e"

export PYTHONPATH="$HOME/GLMToGCompare2/"

SCRIPT_DIR="$(dirname "$0")"
# Define input variables for the Python program
ENCODER_MODELCARD="plenz/GLM-flan-t5-large"
GENERATOR_MODELCARD="google/flan-t5-large"
TRAIN_FILE="$HOME/GLMToGCompare2/data/preprocessed/trex-train-kilt.jsonl"
SAVE_LOCATION="$HOME/GLMToGCompare2/saved_models/trex/flan-t5-large"
PROBLEM_TYPE="classification"
GLM_TYPE="global"
DEVICE="cuda"
BATCH_SIZE=256
OPTIMIZER="AdamW"
LEARNING_RATE="1e-4"
NUM_EPOCHS=5
EARLY_STOPPING=2
NEIGHBORHOOD_SIZE=10
EVAL_FILE="$HOME/GLMToGCompare2/data/preprocessed/trex-dev-kilt.jsonl"
CHECKPOINTING_INTERVAL=500

# Path to the Python program
PYTHON_PROGRAM="$HOME/GLMToGCompare2/GraphLanguageModel/train_glm.py"

# Start the Python program with inputs
echo starting python3 "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"
accelerate launch --multi_gpu --mixed_precision=no --num_processes=10 "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"