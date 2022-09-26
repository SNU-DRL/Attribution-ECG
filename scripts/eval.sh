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

(
gpu=12
method=saliency
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 


method=integrated_gradients
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 
) &
(
gpu=11
method=input_gradient
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 

method=lrp
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 
) &
(
gpu=10
method=lime
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 

method=kernel_shap
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 
) &
(
gpu=9
method=deep_lift
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 

method=deep_lift_shap
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 
) &
(
gpu=8
method=gradcam
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 
) &
(
gpu=7
method=guided_gradcam
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 
) &
(
gpu=6
method=feature_ablation
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --correct --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.9 --gpu $gpu --attr_method $method 
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method --absolute
python evaluate_attributions.py --prob_thres 0.8 --gpu $gpu --attr_method $method 
) 