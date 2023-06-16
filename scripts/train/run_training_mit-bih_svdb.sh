# Dataset
DATASET=mit-bih_svdb
DATASET_PATH=./dataset/data/$DATASET.pkl

# Model
MODEL=resnet18_7

# Hyperparameters
BATCH_SIZE=32
LEARNING_RATE=1e-2
WEIGHT_DECAY=1e-4
EPOCHS=20

# Settings
GPU_NUM=0

RESULT_BASE_DIR='results_training/'$DATASET'_'$MODEL'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS

for SEED in 1 2 3 4 5
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

python process_training_results.py $RESULT_BASE_DIR
