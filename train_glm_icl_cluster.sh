#!/bin/bash
#SBATCH --mail-user=ra443@stud.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=glm-train
#SBATCH --mem=20gb
#SBATCH --partition=afkm
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --output=output_%j.txt
#SBATCH -e error_%j.txt

echo prepare python environment
export PYTHONPATH="/home/students/schwenke/GLMToGCompare2/"
export TOKENIZERS_PARALLELISM=true
source /home/students/schwenke/GLMToGCompare2/.venv/bin/activate

SCRATCH_PATH=/scratch/schwenke
srun cp -r /home/students/schwenke/GLMToGCompare2/data $SCRATCH_PATH

TRITON_CACHE_DIR=$SCRATCH_PATH
srun export TRITON_CACHE_DIR=$TRITON_CACHE_DIR

# Default values
ENCODER_MODELCARD="plenz/GLM-flan-t5-large"
GENERATOR_MODELCARD="google/flan-t5-large"
TRAIN_FILE="$SCRATCH_PATH/data/preprocessed/trex-train-kilt.jsonl"
SAVE_LOCATION="$SCRATCH_PATH/saved_models/trex/flan-t5-large"
PROBLEM_TYPE="classification"
GLM_TYPE="global"
BATCH_SIZE=64
OPTIMIZER="AdamW"
LEARNING_RATE="1e-4"
NUM_EPOCHS=5
EARLY_STOPPING=2
NEIGHBORHOOD_SIZE=10
EVAL_FILE="$SCRATCH_PATH/data/preprocessed/trex-dev-kilt.jsonl"
CHECKPOINTING_INTERVAL=500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e)
            ENCODER_MODELCARD="$2"
            shift 2
            ;;
        -g)
            GENERATOR_MODELCARD="$2"
            shift 2
            ;;
        -t)
            TRAIN_FILE="$2"
            shift 2
            ;;
        -s)
            SAVE_LOCATION="$2"
            shift 2
            ;;
        -p)
            PROBLEM_TYPE="$2"
            shift 2
            ;;
        -l)
            GLM_TYPE="$2"
            shift 2
            ;;
        -b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -o)
            OPTIMIZER="$2"
            shift 2
            ;;
        -a)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -n)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -s)
            EARLY_STOPPING="$2"
            shift 2
            ;;
        -k)
            NEIGHBORHOOD_SIZE="$2"
            shift 2
            ;;
        -d)
            EVAL_FILE="$2"
            shift 2
            ;;
        -c)
            CHECKPOINTING_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Path to the Python program
PYTHON_PROGRAM="/home/students/schwenke/GLMToGCompare2/GraphLanguageModel/train_glm.py"

# Start the Python program with inputs
echo "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"
srun python "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
        -gt "$GLM_TYPE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
        -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL"
