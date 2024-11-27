#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --time=120
#SBATCH --mem=64gb
#SBATCH --job-name=trex-large
#SBATCH --gres=gpu:1
#SBATCH -o "./trex-large.o"
#SBATCH --error="./trex-large.e"

# Extract compressed input dataset on local SSD
tar -C $TMPDIR/ -xvzf $(ws_find data-fast)/data.tgz

export PYTHONPATH="$HOME/GLMToGCompare2/"

SCRIPT_DIR="$(dirname "$0")"
# Define input variables for the Python program
ENCODER_MODELCARD="plenz/GLM-flan-t5-large"
GENERATOR_MODELCARD="google/flan-t5-large"
TRAIN_FILE="$TMPDIR/data/preprocessed/trex-train-kilt.jsonl"
SAVE_LOCATION="$TMPDIR/saved_models/trex/flan-t5-large"
PROBLEM_TYPE="classification"
GLM_TYPE="global"
DEVICE="cuda"
BATCH_SIZE=32
OPTIMIZER="AdamW"
LEARNING_RATE="1e-4"
NUM_EPOCHS=5
EARLY_STOPPING=2
NEIGHBORHOOD_SIZE=10
EVAL_FILE="$TMPDIR/data/preprocessed/trex-dev-kilt.jsonl"
CHECKPOINTING_INTERVAL=500

# activate venv
source $HOME/GLMToGCompare2/.venv/bin/activate
# Path to the Python program
PYTHON_PROGRAM="$HOME/GLMToGCompare2/GraphLanguageModel/train_glm.py"

# Start the Python program with inputs
echo starting python3 "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"
accelerate launch --mixed_precision=bf16 --dynamo_backend=cudagraphs  "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"

rsync -av $SAVE_LOCATION $(ws_find data-fast)/saved_models-${SLURM_JOB_ID}/