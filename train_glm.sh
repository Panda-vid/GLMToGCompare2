#!/bin/bash
#SBATCH --ntasks=256
#SBATCH --time=120
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1

# set up enviroment for cuda
module load devel/cuda/12.4
# Extract compressed input dataset on local SSD
echo extracting data files onto $TMPDIR
tar -C $TMPDIR/ -xvzf $(ws_find data-fast)/data.tgz

export PYTHONPATH="$HOME/GLMToGCompare2/"
export TOKENIZERS_PARALLELISM=true

# Define input variables for the Python program
ENCODER_MODELCARD=""
if [ -e "$ENCODER_MODELCARD" ]; then
        ENCODER_MODELCARD="plenz/GLM-flan-t5-large"
fi
GENERATOR_MODELCARD=""
if [ -g "$GENERATOR_MODELCARD" ]; then
        GENERATOR_MODELCARD="google/flan-t5-large"
fi
TRAIN_FILE=""
if [ -t "$TRAIN_FILE" ]; then
        TRAIN_FILE="$TMPDIR/data/preprocessed/trex-train-kilt.jsonl"
fi
SAVE_LOCATION=""
if [ -s "$SAVE_LOCATION"]; then
        SAVE_LOCATION="$TMPDIR/saved_models/trex/flan-t5-large"
fi
PROBLEM_TYPE=""
if [ -p "$PROBLEM_TYPE"]; then
        PROBLEM_TYPE="classification"
fi
GLM_TYPE=""
if [ -l "$GLM_TYPE"]; then
        GLM_TYPE="global"
fi
BATCH_SIZE=""
if [ -b "$BATCH_SIZE"]; then
        BATCH_SIZE=128
fi
OPTIMIZER=""
if [ -o "$OPTIMIZER"]; then
        OPTIMIZER="AdamW"
fi
LEARNING_RATE=""
if [ -a "$LEARNING_RATE"]; then
        LEARNING_RATE="1e-4"
fi
NUM_EPOCHS=""
if [ -n "$NUM_EPOCHS"]; then
        NUM_EPOCHS=5
fi
EARLY_STOPPING=""
if [ -s "$EARLY_STOPPING"]; then
        EARLY_STOPPING=2
fi
NEIGHBORHOOD_SIZE=""
if [ -k "$NEIGHBORHOOD_SIZE"]; then
        NEIGHBORHOOD_SIZE=10
fi
EVAL_FILE=""
if [ -d "$EVAL_FILE"]; then
        EVAL_FILE="$TMPDIR/data/preprocessed/trex-dev-kilt.jsonl"
fi
CHECKPOINTING_INTERVAL=""
if [ -c "$CHEKCPOINTING_INTERVAL"]; then
        CHECKPOINTING_INTERVAL=500
fi

# activate venv
source $HOME/GLMToGCompare2/.venv/bin/activate
# Path to the Python program
PYTHON_PROGRAM="$HOME/GLMToGCompare2/GraphLanguageModel/train_glm.py"

# Start the Python program with inputs
echo accelerate launch "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"
accelerate launch --mixed_precision=bf16 --dynamo_backend=cudagraphs --num_machines 1  "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"

rsync -av $TMPDIR/saved_models $(ws_find data-fast)/saved_models-${SLURM_JOB_ID}/