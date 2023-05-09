# Dataset
DATASET=icentia11k
DATASET_PATH=./data/12000_btype_new.pkl

# Model
MODEL=resnet18_7
BATCH_SIZE=32
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
EPOCHS=20
BASE_DIR=$DATASET'_'$MODEL'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS

# Feature attribution methods
PROB_THRESHOLD=0.9
ATTR_METHOD=gradcam
N_SAMPLES=500

# Evaluation metrics
DEG_WINDOW_SIZE=16

# Settings
GPU_NUM=0

mkdir results_eval

for SEED in 1 2 3 4 5
do
    MODEL_DIR=$BASE_DIR'_seed'$SEED
    MODEL_PATH=results/$MODEL_DIR/model_last.pt

    RESULT_DIR=results_eval/$MODEL_DIR/$ATTR_METHOD
    python evaluate_attribution.py \
        --dataset $DATASET \
        --dataset_path $DATASET_PATH \
        --model_path $MODEL_PATH \
        --prob_threshold $PROB_THRESHOLD \
        --attr_method $ATTR_METHOD \
        --n_samples $N_SAMPLES \
        --deg_window_size $DEG_WINDOW_SIZE \
        --gpu_num $GPU_NUM \
        --seed $SEED \
        --result_dir $RESULT_DIR

    RESULT_DIR=results_eval/$MODEL_DIR/$ATTR_METHOD'_absolute'
    python evaluate_attribution.py \
        --dataset $DATASET \
        --dataset_path $DATASET_PATH \
        --model_path $MODEL_PATH \
        --prob_threshold $PROB_THRESHOLD \
        --attr_method $ATTR_METHOD \
        --absolute \
        --n_samples $N_SAMPLES \
        --deg_window_size $DEG_WINDOW_SIZE \
        --gpu_num $GPU_NUM \
        --seed $SEED \
        --result_dir $RESULT_DIR
done

python process_evaluation_results.py results_eval/$BASE_DIR