# Dataset
DATASET=mitdb

# Model
MODEL=resnet18_7
BATCH_SIZE=32
LEARNING_RATE=5e-2
WEIGHT_DECAY=1e-4
EPOCHS=20
BASE_DIR=$DATASET'_'$MODEL'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS

# Evaluation metrics
PATCH_SIZE=24 # for region_perturbation

# Settings
GPU_NUM=7

for ATTR_METHOD in gradcam random_baseline input_gradient saliency guided_backprop integrated_gradients deep_lift deep_shap lrp guided_gradcam lime kernel_shap
do
    for SEED in 1 2 3 4 5
    do
        MODEL_DIR=$BASE_DIR'_seed'$SEED
        MODEL_PATH=results_training/$MODEL_DIR/model_last.pt
        ATTR_DIR=results_attribution/$MODEL_DIR/$ATTR_METHOD

        RESULT_DIR=results_evaluation_region_perturbation/$MODEL_DIR/$ATTR_METHOD
        ###

        EVAL_METRIC=region_perturbation
        PERTURB_ORDER=lerf # for region_perturbation
        
        python evaluate_attribution.py \
            --attr_dir $ATTR_DIR \
            --model_path $MODEL_PATH \
            --eval_metric $EVAL_METRIC \
            --patch_size $PATCH_SIZE \
            --perturb_order $PERTURB_ORDER \
            --gpu_num $GPU_NUM \
            --seed $SEED \
            --result_dir $RESULT_DIR

        python evaluate_attribution.py \
            --attr_dir $ATTR_DIR \
            --model_path $MODEL_PATH \
            --eval_metric $EVAL_METRIC \
            --patch_size $PATCH_SIZE \
            --perturb_order $PERTURB_ORDER \
            --absolute \
            --gpu_num $GPU_NUM \
            --seed $SEED \
            --result_dir $RESULT_DIR
        ###

        EVAL_METRIC=region_perturbation
        PERTURB_ORDER=morf # for region_perturbation
        
        python evaluate_attribution.py \
            --attr_dir $ATTR_DIR \
            --model_path $MODEL_PATH \
            --eval_metric $EVAL_METRIC \
            --patch_size $PATCH_SIZE \
            --perturb_order $PERTURB_ORDER \
            --gpu_num $GPU_NUM \
            --seed $SEED \
            --result_dir $RESULT_DIR

        python evaluate_attribution.py \
            --attr_dir $ATTR_DIR \
            --model_path $MODEL_PATH \
            --eval_metric $EVAL_METRIC \
            --patch_size $PATCH_SIZE \
            --perturb_order $PERTURB_ORDER \
            --absolute \
            --gpu_num $GPU_NUM \
            --seed $SEED \
            --result_dir $RESULT_DIR
        ###
    done
done

python analysis/summarize_evaluation.py results_evaluation_region_perturbation/$BASE_DIR