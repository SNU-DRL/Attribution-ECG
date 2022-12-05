# attr_methods_list = [
#     'saliency', 
#     'integrated_gradients', 
#     'input_gradient', 
#     'lrp', 
#     'lime', 
#     'kernel_shap', 
#     'deep_lift',
#     'deep_lift_shap',
#     'gradcam',
#     'guided_gradcam',
#     'feature_ablation',
# ]

gpu=$1
method=$2
model_path=$3
results_path=$4

python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --model_path $model_path --results_path $results_dir --absolute &
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --model_path $model_path --results_path $results_dir 

