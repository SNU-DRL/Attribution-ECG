import math

import numpy as np
import torch
from captum.attr import (LRP, DeepLift, DeepLiftShap, GuidedBackprop,
                         GuidedGradCam, InputXGradient, IntegratedGradients,
                         KernelShap, LayerAttribution, LayerGradCam, Lime,
                         Saliency)

ATTRIBUTION_METHODS = {
    "random_baseline": None,
    "saliency": Saliency,
    "input_gradient": InputXGradient,
    "guided_backprop": GuidedBackprop,
    "integrated_gradients": IntegratedGradients,
    "deep_lift": DeepLift,
    "deep_shap": DeepLiftShap,
    "lrp": LRP,
    "lime": Lime,
    "kernel_shap": KernelShap,
    "gradcam": LayerGradCam,
    "guided_gradcam": GuidedGradCam,
}


class Attribution:
    """
    Apply feature attribution method to a sample (x, y)
    """

    def __init__(self, model, attr_method, n_samples, feature_mask_size, len_x, num_leads, device):
        self.model = model
        self.attr_method = attr_method
        if attr_method == "random_baseline":
            self.attr_func = None
        elif "gradcam" in attr_method:
            self.attr_func = ATTRIBUTION_METHODS[attr_method](
                model, model.backbone.layer4
            )
        else:
            self.attr_func = ATTRIBUTION_METHODS[attr_method](model)
        self.n_samples = n_samples
        self.feature_mask_size = feature_mask_size
        self.len_x = len_x
        self.num_leads = num_leads
        self.device = device
        
        num_feature_mask_repeats = math.ceil(len_x / feature_mask_size)
        feature_mask_rows = [np.repeat(np.arange(num_feature_mask_repeats*i, num_feature_mask_repeats*(i+1)), feature_mask_size) for i in range(self.num_leads)]
        feature_mask = np.stack(feature_mask_rows).reshape(1,1,num_leads,-1)[:,:,:,:len_x]
        self.feature_mask = torch.tensor(feature_mask, device=device)

    def apply(self, x, y) -> np.ndarray:
        if self.attr_method == "random_baseline":
            attr_x = np.random.randn(*x.shape)
            return attr_x
        elif self.attr_method == "saliency":
            attr_x = self.attr_func.attribute(x, target=y, abs=False)
        elif self.attr_method in ["lime", "kernel_shap"]:
            attr_x = self.attr_func.attribute(x, target=y, n_samples=self.n_samples, feature_mask=self.feature_mask)
        elif self.attr_method == "deep_shap":
            attr_x = self.attr_func.attribute(
                x,
                target=y,
                baselines=torch.randn(
                    [self.n_samples, *list(x.shape[1:])], device=self.device
                ),
            )
        else:
            attr_x = self.attr_func.attribute(x, target=y)

        # Interpolation for GradCAM
        if self.attr_method == "gradcam":
            attr_x = LayerAttribution.interpolate(attr_x, x.shape[-2:])

        return attr_x.detach().cpu().numpy()
