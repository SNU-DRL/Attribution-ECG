import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import (LRP, DeepLift, DeepLiftShap, FeatureAblation,
                         GuidedBackprop, GuidedGradCam, InputXGradient,
                         IntegratedGradients, KernelShap, LayerAttribution,
                         LayerGradCam, Lime, Saliency)


def apply_attr_method(
    model,
    attr_method,
    input_tensor,
    n_samples=200,
    absolute=False,
):
    ## Select target (predicted label)
    yhat = model(input_tensor)
    softmax_yhat = F.softmax(yhat, dim=1)
    prediction_score, pred_label_idx = torch.topk(softmax_yhat, 1)

    attribution_dict = {
        "saliency": Saliency,
        "input_gradient": InputXGradient,
        "guided_backprop": GuidedBackprop,
        "integrated_gradients": IntegratedGradients,
        "deep_lift": DeepLift,
        "deep_lift_shap": DeepLiftShap,
        "lrp": LRP,
        "lime": Lime,
        "kernel_shap": KernelShap,
        "gradcam": LayerGradCam,
        "guided_gradcam": GuidedGradCam,
    }

    pred_label_idx.squeeze_()
    # print("pred_label_idx.shape", pred_label_idx.shape)

    ## Load attribution function
    if "gradcam" in attr_method:
        attr_func = attribution_dict[attr_method](model, model.backbone.layer4)
    else:
        attr_func = attribution_dict[attr_method](model)

    ## Conduct attribution method
    if attr_method in ["lime", "kernel_shap"]:
        attr_x = attr_func.attribute(
            input_tensor, target=pred_label_idx, n_samples=n_samples
        )
    elif attr_method == "deep_lift_shap":
        attr_x = attr_func.attribute(
            input_tensor,
            target=pred_label_idx,
            baselines=torch.randn([n_samples] + list(input_tensor.shape[1:])),
        )
    else:
        attr_x = attr_func.attribute(input_tensor, target=pred_label_idx)

    ## Interpolation for GradCAM
    if attr_method == "gradcam":
        attr_x = LayerAttribution.interpolate(attr_x, (1, input_tensor.shape[-1]))

    ## Absolute values of attribution
    if absolute:
        attr_x = torch.abs(attr_x)

    return attr_x


def localization_score(attr_x, true_label, boundaries_per_label):
    N = 0
    true_idx = []
    for boundary in boundaries_per_label[true_label]:
        N += boundary[1] - boundary[0]
        true_idx += list(np.arange(*boundary))
    attr_topN = np.argsort(attr_x)[-N:]
    true_idx = set(true_idx)
    pred_idx = set(attr_topN)

    iou = len(pred_idx & true_idx) / len(pred_idx | true_idx)
    return iou


def pointing_game(attr_x, true_label, boundaries_per_label):
    attr_top1 = np.argmax(attr_x)

    correct = False
    for boundary in boundaries_per_label[true_label]:
        correct = correct or (attr_top1 in range(*boundary))
    return correct


def degradation_score(
    attr_x,
    true_label,
    x,
    model,
    device,
    method="mean",
    window_size=16,
):
    """
    methods
    - zero: replace the erased part with 0
    - mean: replace the erased part with the mean of each edge
    - linear: replace the erased part with the linear interpolation of each edge
    """
    truncate_idx = len(x) % window_size
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
        degraded_x = degrade(degraded_x, window_idx, method, window_size)
        LeRF_x_list.append(torch.tensor(degraded_x).reshape(1, 1, 1, -1))

    degraded_x = np.copy(x)
    for window_idx in MoRF_rank:
        degraded_x = degrade(degraded_x, window_idx, method, window_size)
        MoRF_x_list.append(torch.tensor(degraded_x).reshape(1, 1, 1, -1))

    LeRF_x = torch.cat(LeRF_x_list, 0).to(device)
    MoRF_x = torch.cat(MoRF_x_list, 0).to(device)

    with torch.no_grad():
        LeRF_prob = F.softmax(model(LeRF_x), dim=1)[:, true_label]
        MoRF_prob = F.softmax(model(MoRF_x), dim=1)[:, true_label]

    return LeRF_prob.tolist(), MoRF_prob.tolist()


def degrade(x, idx, method, window_size):
    x = x.reshape(-1, window_size)
    if idx == 0:
        left_end = x[idx][0]
        right_end = x[idx + 1][0]
    elif idx == len(x) - 1:
        left_end = x[idx - 1][-1]
        right_end = x[idx][-1]
    else:
        left_end = x[idx - 1][-1]
        right_end = x[idx + 1][0]

    if method == "zero":
        x[idx] = np.zeros(window_size)
    elif method == "mean":
        x[idx] = np.full(window_size, x[idx].mean())
    elif method == "linear":
        x[idx] = np.linspace(left_end, right_end, window_size)
    elif method == "gaussian":
        x[idx] = np.random.randn(window_size)
    elif method == "gaussian_plus":
        x[idx] = x[idx] + np.random.randn(window_size)

    x = x.reshape(-1)
    return x
