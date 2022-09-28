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

results_dir='results_eval_2'

(
gpu=12
method=saliency
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=11
method=integrated_gradients
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=10
method=input_gradient
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=9
method=lrp
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=8
method=lime
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=7
method=kernel_shap
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=6
method=deep_lift
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=5
method=deep_lift_shap
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=4
method=gradcam
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=3
method=guided_gradcam
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) &
(
gpu=2
method=feature_ablation
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute 
python evaluate_attributions.py --results_path $results_dir --prob_thres 0.9 --gpu $gpu --attr_method $method 
) 
