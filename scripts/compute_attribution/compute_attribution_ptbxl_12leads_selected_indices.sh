# Dataset
DATASET=ptbxl
DATASET_PATH=./ptb-xl/data/data_ptbxl/$DATASET'_12leads.pkl'

# Model
MODEL=resnet18_7
BATCH_SIZE=32
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
EPOCHS=20
BASE_DIR=$DATASET'_12leads_'$MODEL'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS

# Distribution directory
DIST_DIR="results_ptbxl/dist_12leads"

# Feature attribution methods
# PROB_THRESHOLD=0.75
# ATTR_METHOD=gradcam
# N_SAMPLES=500
FEATURE_MASK_SIZE=32

# Settings
GPU_NUM=1

for ATTR_METHOD in guided_gradcam lime
do
    if [ "$ATTR_METHOD" = "deep_shap" ]; then
        N_SAMPLES=80
   else
        N_SAMPLES=5000
    fi
    for SEED in 0
    do  
        for TARGET_LABEL in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
        do
            MODEL_DIR=$BASE_DIR'_seed'$SEED
            MODEL_PATH=results_ptbxl/results_training/$MODEL_DIR/model_last.pt

            RESULT_DIR=results_ptbxl/results_attribution_selected/$MODEL_DIR/$ATTR_METHOD/$TARGET_LABEL
            python compute_attribution_ptbxl_selected.py \
                --dataset $DATASET \
                --dataset_path $DATASET_PATH \
                --dist_dir $DIST_DIR \
                --model_path $MODEL_PATH \
                --attr_method $ATTR_METHOD \
                --n_samples $N_SAMPLES \
                --feature_mask_size $FEATURE_MASK_SIZE \
                --gpu_num $GPU_NUM \
                --seed $SEED \
                --multi_label \
                --target_label $TARGET_LABEL \
                --result_dir $RESULT_DIR
        done
    done
done