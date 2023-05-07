import numpy as np
import torch
import torch.nn.functional as F
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

    def __init__(self, model, attr_method, absolute=False, n_samples=200):
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
        self.absolute = absolute
        self.n_samples = n_samples

    def apply(self, x, y) -> np.ndarray:
        if self.attr_method == "random_baseline":
            attr_x = np.random.randn(*x.shape)
            return attr_x
        elif self.attr_method == "saliency":
            attr_x = self.attr_func.attribute(x, target=y, abs=self.absolute)
        elif self.attr_method in ["lime", "kernel_shap"]:
            attr_x = self.attr_func.attribute(x, target=y, n_samples=self.n_samples)
        elif self.attr_method == "deep_shap":
            attr_x = self.attr_func.attribute(
                x,
                target=y,
                baselines=torch.randn(
                    [250, *list(x.shape[1:])], device=x.device
                ),
            )
        else:
            attr_x = self.attr_func.attribute(x, target=y)

        # Interpolation for GradCAM
        if self.attr_method == "gradcam":
            attr_x = LayerAttribution.interpolate(attr_x, (1, x.shape[-1]))

        # Use absolute values of attribution
        if self.absolute:
            attr_x = torch.abs(attr_x)

        return attr_x.detach().cpu().numpy()


def localization_score(attr_x, y, beat_spans):
    N = 0
    true_idx = []
    for span in beat_spans[y]:
        N += span[1] - span[0]
        true_idx += list(np.arange(*span))
    attr_topN = np.argsort(attr_x)[-N:]
    true_idx = set(true_idx)
    pred_idx = set(attr_topN)

    iou = len(pred_idx & true_idx) / len(pred_idx | true_idx)
    return iou


def pointing_game(attr_x, y, beat_spans):
    attr_top1 = np.random.choice(np.flatnonzero(attr_x == np.max(attr_x))) # random tie-breaking argmax
    is_correct = False
    for span in beat_spans[y]:
        is_correct = is_correct or (attr_top1 in range(*span))
    return is_correct


def degradation_score(
    model,
    x,
    attr_x,
    y,
    device,
    perturbation="mean",
    window_size=16,
):
    """
    perturbation
    - zero: replace the window with 0
    - mean: fill the window with the mean value of the window
    - linear: replace the window with the linear interpolation of both edges
    - gaussian: replace the window with Gaussian noise
    - gaussian_pluse: add a Gaussian noise to the window
    """
    truncate_idx = len(x) % window_size
    if truncate_idx > 0:
        x, attr_x = x[:-truncate_idx], attr_x[:-truncate_idx]
    attr_x = attr_x.reshape(-1, window_size)

    attr_window_score = attr_x.sum(1)
    LeRF_rank = np.argsort(attr_window_score)
    MoRF_rank = LeRF_rank[::-1]

    LeRF_x_list = []
    MoRF_x_list = []

    LeRF_x_list.append(torch.tensor(x).reshape(1, 1, 1, -1))
    MoRF_x_list.append(torch.tensor(x).reshape(1, 1, 1, -1))

    degraded_x = np.copy(x)
    for window_idx in LeRF_rank:
        degraded_x = degrade(degraded_x, window_idx, perturbation, window_size)
        LeRF_x_list.append(torch.tensor(degraded_x).reshape(1, 1, 1, -1))

    degraded_x = np.copy(x)
    for window_idx in MoRF_rank:
        degraded_x = degrade(degraded_x, window_idx, perturbation, window_size)
        MoRF_x_list.append(torch.tensor(degraded_x).reshape(1, 1, 1, -1))

    LeRF_x = torch.cat(LeRF_x_list, 0).to(device)
    MoRF_x = torch.cat(MoRF_x_list, 0).to(device)

    with torch.no_grad():
        LeRF_prob = F.softmax(model(LeRF_x), dim=1)[:, y]
        MoRF_prob = F.softmax(model(MoRF_x), dim=1)[:, y]

    return LeRF_prob.tolist(), MoRF_prob.tolist()


def degrade(x, idx, perturbation, window_size):
    x = x.reshape(-1, window_size)

    if perturbation == "zero":
        x[idx] = np.zeros(window_size)
    elif perturbation == "mean":
        x[idx] = np.full(window_size, x[idx].mean())
    elif perturbation == "linear":
        if idx == 0:
            left_end = x[idx][0]
            right_end = x[idx + 1][0]
        elif idx == len(x) - 1:
            left_end = x[idx - 1][-1]
            right_end = x[idx][-1]
        else:
            left_end = x[idx - 1][-1]
            right_end = x[idx + 1][0]
        x[idx] = np.linspace(left_end, right_end, window_size)
    elif perturbation == "gaussian":
        x[idx] = np.random.randn(window_size)
    elif perturbation == "gaussian_plus":
        x[idx] = x[idx] + np.random.randn(window_size)

    x = x.reshape(-1)
    return x
