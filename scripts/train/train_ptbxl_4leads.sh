# Dataset
DATASET=ptbxl
DATASET_PATH=./ptb-xl/data/data_ptbxl/$DATASET'_4leads.pkl'

# Model
MODEL=resnet18_7

# Hyperparameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
EPOCHS=20

# Settings
GPU_NUM=0

RESULT_BASE_DIR='results_training/'$DATASET'_4leads_'$MODEL'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS

for SEED in 0
do
    RESULT_DIR=$RESULT_BASE_DIR'_seed'$SEED

    python train.py \
        --dataset $DATASET \
        --dataset_path $DATASET_PATH \
        --model $MODEL \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --epochs $EPOCHS \
        --gpu_num $GPU_NUM \
        --seed $SEED \
        --result_dir $RESULT_DIR
done

