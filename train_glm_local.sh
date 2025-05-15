export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true

# Default values
ENCODER_MODELCARD="/GLM-flan-t5-base"
GENERATOR_MODELCARD="google/flan-t5-base"
TRAIN_FILE="./data/preprocessed/structured_zeroshot-train-kilt.jsonl"
SAVE_LOCATION="./saved_models/model_compare/flan-t5-base"
PROBLEM_TYPE="classification"
GLM_TYPE="global"
BATCH_SIZE=16
OPTIMIZER="AdamW"
DEVICE="cuda"
LEARNING_RATE="1e-8"
NUM_EPOCHS=5
EARLY_STOPPING=2
NEIGHBORHOOD_SIZE=10
EVAL_FILE="./data/preprocessed/structured_zeroshot-dev-kilt.jsonl"
CHECKPOINTING_INTERVAL=2500

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

# activate venv
source ./.venv/bin/activate
# Path to the Python program
PYTHON_PROGRAM=./GraphLanguageModel/train_glm.py

# Start the Python program with inputs
for i in $(seq 1 2);
do
    echo python "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
            -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
            -ns "$NEIGHBORHOOD_SIZE" -ef "$EVAL_FILE" -c "$CHECKPOINTING_INTERVAL" -l --trace_loss --gradient_checkpointing
    timeout 4200 python "$PYTHON_PROGRAM" "$ENCODER_MODELCARD" "$GENERATOR_MODELCARD" "$TRAIN_FILE" "$SAVE_LOCATION" -pt "$PROBLEM_TYPE" \
            -gt "$GLM_TYPE" -d "$DEVICE" -b "$BATCH_SIZE" -o "$OPTIMIZER" -lr "$LEARNING_RATE" -ne "$NUM_EPOCHS" -es "$EARLY_STOPPING" \
            -ns "$NEIGHBORHOOD_SIZE" -c "$CHECKPOINTING_INTERVAL" --trace_loss --gradient_checkpointing
done
