import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.attr import Saliency, IntegratedGradients, DeepLiftShap, Lime, LayerGradCam, GuidedGradCam, InputXGradient, KernelShap, FeatureAblation, LayerAttribution, LRP, DeepLift
from captum.attr import visualization as viz

def compute_attr_x(model, attr_method, input_tensor, n_samples=200, absolute=False):
    ## Select target (predicted label)
    yhat = model(input_tensor)
    softmax_yhat = F.softmax(yhat, dim=1)
    prediction_score, pred_label_idx = torch.topk(softmax_yhat, 1)

    attribution_dict = {"saliency": Saliency(model),
                        "integrated_gradients": IntegratedGradients(model),
                        "input_gradient": InputXGradient(model),
                        "lrp": LRP(model),
                        "lime": Lime(model),
                        "kernel_shap": KernelShap(model),
                        "deep_lift": DeepLift(model),
                        "deep_lift_shap": DeepLiftShap(model),
                        "gradcam": LayerGradCam(model, model.layer4),
                        "guided_gradcam": GuidedGradCam(model, model.layer4),
                        "feature_ablation": FeatureAblation(model),
                        }

    ## Load attribution function
    attr_func = attribution_dict[attr_method]
    
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