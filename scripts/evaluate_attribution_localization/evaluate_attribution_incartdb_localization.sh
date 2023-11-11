# Dataset
DATASET=incartdb

# Model
MODEL=resnet18_7
BATCH_SIZE=32
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
EPOCHS=20
BASE_DIR=$DATASET'_'$MODEL'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS

# Settings
GPU_NUM=0

for EVAL_METRIC in attribution_localization auc pointing_game relevance_rank_accuracy
do
    for ATTR_METHOD in random_baseline input_gradient saliency guided_backprop integrated_gradients deep_lift deep_shap lrp gradcam guided_gradcam lime kernel_shap
    do
        for SEED in 1 2 3 4 5
        do
            MODEL_DIR=$BASE_DIR'_seed'$SEED
            MODEL_PATH=results_for_paper/results_training/$MODEL_DIR/model_last.pt
            ATTR_DIR=results_for_paper/results_attribution/$MODEL_DIR/$ATTR_METHOD

            RESULT_DIR=results_evaluation_localization/$MODEL_DIR/$ATTR_METHOD
            python evaluate_attribution.py \
                --attr_dir $ATTR_DIR \
                --model_path $MODEL_PATH \
                --eval_metric $EVAL_METRIC \
                --gpu_num $GPU_NUM \
                --seed $SEED \
                --result_dir $RESULT_DIR

            python evaluate_attribution.py \
                --attr_dir $ATTR_DIR \
                --model_path $MODEL_PATH \
                --eval_metric $EVAL_METRIC \
                --absolute \
                --gpu_num $GPU_NUM \
                --seed $SEED \
                --result_dir $RESULT_DIR
        done
    done
done

python analysis/summarize_evaluation.py results_evaluation_localization/$BASE_DIR