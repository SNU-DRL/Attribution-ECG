import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import (LRP, DeepLift, DeepLiftShap, FeatureAblation,
                         GuidedGradCam, InputXGradient, IntegratedGradients,
                         KernelShap, LayerAttribution, LayerGradCam, Lime,
                         Saliency)
from captum.attr import visualization as viz


def compute_attr_x(model, attr_method, input_tensor, n_samples=200, absolute=False):
    ## Select target (predicted label)
    yhat = model(input_tensor)
    softmax_yhat = F.softmax(yhat, dim=1)
    prediction_score, pred_label_idx = torch.topk(softmax_yhat, 1)
    pred_label_idx.squeeze_()
    # print("pred_label_idx.shape", pred_label_idx.shape)

    attribution_dict = {"saliency": Saliency,
                        "integrated_gradients": IntegratedGradients,
                        "input_gradient": InputXGradient,
                        "lrp": LRP,
                        "lime": Lime,
                        "kernel_shap": KernelShap,
                        "deep_lift": DeepLift,
                        "deep_lift_shap": DeepLiftShap,
                        "gradcam": LayerGradCam,
                        "guided_gradcam": GuidedGradCam,
                        "feature_ablation": FeatureAblation,
                        }

    ## Load attribution function
    if 'gradcam' in attr_method:
        attr_func = attribution_dict[attr_method](model, model.layer4)
    else:
        attr_func = attribution_dict[attr_method](model)
    
    ## Conduct attribution method
    if attr_method in ["lime", "kernel_shap"]:
        attr_x = attr_func.attribute(input_tensor, target=pred_label_idx, n_samples=n_samples)
    elif attr_method == "deep_lift_shap":
        attr_x = attr_func.attribute(input_tensor, target=pred_label_idx, baselines=torch.randn([n_samples] + list(input_tensor.shape[1:])).cuda())
    else:
        attr_x = attr_func.attribute(input_tensor, target=pred_label_idx)

    ## Interpolation for GradCAM
    if attr_method == "gradcam":
        attr_x = LayerAttribution.interpolate(attr_x, (1, input_tensor.shape[-1]))
    
    ## Absolute values of attribution
    if absolute:
        attr_x = torch.abs(attr_x)

    return attr_x


def to_np(x):
    return x.detach().cpu().numpy()