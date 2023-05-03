# Dataset
DATASET=mit-bih
DATASET_PATH=./data/mit-bih.pkl

# Model
MODEL=resnet18_7

# Hyperparameters
BATCH_SIZE=128
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
EPOCHS=30

# Settings
GPU_NUM=0

RESULT_BASE_DIR='results/'$DATASET'_'$MODEL'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS

mkdir results

for SEED in 0 1 2 3 4
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
