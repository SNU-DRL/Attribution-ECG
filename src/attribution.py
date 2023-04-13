import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import (
    LRP,
    DeepLift,
    DeepLiftShap,
    FeatureAblation,
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    KernelShap,
    LayerAttribution,
    LayerGradCam,
    Lime,
    Saliency,
)

from utils import get_boundaries_by_label, flatten_raw_label


def get_attribution(
    model,
    attr_method,
    input_tensor,
    n_samples=200,
    feature_mask_window=16,
    absolute=False,
):
    ## Select target (predicted label)
    yhat = model(input_tensor)
    softmax_yhat = F.softmax(yhat, dim=1)
    prediction_score, pred_label_idx = torch.topk(softmax_yhat, 1)

    attribution_dict = {
        "saliency": Saliency,
        "integrated_gradients": IntegratedGradients,
        "input_gradient": InputXGradient,
        "guided_backprop": GuidedBackprop,
        "lrp": LRP,
        "lime": Lime,
        "kernel_shap": KernelShap,
        "deep_lift": DeepLift,
        "deep_lift_shap": DeepLiftShap,
        "gradcam": LayerGradCam,
        "guided_gradcam": GuidedGradCam,
        "feature_ablation": FeatureAblation,
    }

    pred_label_idx.squeeze_()
    # print("pred_label_idx.shape", pred_label_idx.shape)

    ## Load attribution function
    if "gradcam" in attr_method:
        attr_func = attribution_dict[attr_method](model, model.layer4)
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
    elif attr_method == "feature_ablation":
        feature_mask = torch.cat(
            (
                torch.arange(
                    input_tensor.shape[-1] // feature_mask_window
                ).repeat_interleave(feature_mask_window),
                torch.Tensor(
                    [input_tensor.shape[-1] // feature_mask_window - 1]
                    * (input_tensor.shape[-1] % feature_mask_window)
                ),
            ),
            0,
        ).int()
        feature_mask = feature_mask.view(input_tensor.shape)
        attr_x = attr_func.attribute(
            input_tensor, target=pred_label_idx, feature_mask=feature_mask
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


def evaluate_attribution(
    x,
    model,
    attr_x,
    true_label,
    raw_label,
    degradation_method_list=["mean", "linear", "gaussian_plus"],
):
    """
    x: numpy
    model: torch
    attr_x: numpy
    true_label: str
    raw_label: list
    """
    model.eval()
    x = np.squeeze(x)
    attr_x = np.squeeze(attr_x)

    boundaries_per_label = get_boundaries_by_label(raw_label)
    flat_raw_label = flatten_raw_label(raw_label)
    flat_raw_label = dict(sorted(flat_raw_label.items()))

    loc_metric = localization_score(
        attr_x, true_label, boundaries_per_label, x=x, model=model
    )
    pnt_metric = pointing_game(
        attr_x, true_label, boundaries_per_label, x=x, model=model
    )
    deg_metric = {}
    for method in degradation_method_list:
        deg_metric[method] = degradation_score(
            attr_x, true_label, boundaries_per_label, x=x, model=model, method=method
        )

    return loc_metric, pnt_metric, deg_metric


def localization_score(attr_x, true_label, boundaries_per_label, **kwargs):
    N = 0
    true_idx = []
    for boundary in boundaries_per_label[true_label]:
        N += boundary[1] - boundary[0]
        true_idx += list(np.arange(*boundary))
    attr_topN = np.argsort(attr_x)[-N:]
    true_idx = set(true_idx)
    pred_idx = set(attr_topN)

    iou = len(pred_idx & true_idx) / len(pred_idx | true_idx)

    return {"iou": iou}


def pointing_game(attr_x, true_label, boundaries_per_label, **kwargs):
    attr_top1 = np.argmax(attr_x)

    correct = False
    for boundary in boundaries_per_label[true_label]:
        correct = correct or (attr_top1 in range(*boundary))
    return {"correct": correct}


def degradation_score(
    attr_x,
    true_label,
    boundaries_per_label,
    x,
    model,
    method="mean",
    window_size=16,
    **kwargs
):
    """
    methods
    - zero: replace the erased part with 0
    - mean: replace the erased part with the mean of each edge
    - linear: replace the erased part with the linear interpolation of each edge
    """

    new_x, attr_x = np.copy(x[:-1]), attr_x[:-1]
    attr_x_reshaped = attr_x.reshape(-1, window_size)
    score_per_window = attr_x_reshaped.sum(1)
    LeRF_rank = np.argsort(score_per_window)
    MoRF_rank = LeRF_rank[::-1]

    LeRF_x_list = []
    MoRF_x_list = []

    LeRF_x_list.append(torch.tensor(new_x).reshape(1, 1, 1, -1))
    MoRF_x_list.append(torch.tensor(new_x).reshape(1, 1, 1, -1))

    new_x = np.copy(x[:-1])
    for window_idx in LeRF_rank:
        new_x = degrade(new_x, window_idx, method=method)
        LeRF_x_list.append(torch.tensor(new_x).reshape(1, 1, 1, -1))

    new_x = np.copy(x[:-1])
    for window_idx in MoRF_rank:
        new_x = degrade(new_x, window_idx, method=method)
        MoRF_x_list.append(torch.tensor(new_x).reshape(1, 1, 1, -1))

    LeRF_x = torch.cat(LeRF_x_list, 0)
    MoRF_x = torch.cat(MoRF_x_list, 0)

    with torch.no_grad():
        LeRF_prob = F.softmax(model(LeRF_x), dim=1)[:, true_label]
        MoRF_prob = F.softmax(model(MoRF_x), dim=1)[:, true_label]

    return {"LeRF": LeRF_prob.tolist(), "MoRF": MoRF_prob.tolist()}


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

    replace_values = {
        "zero": np.zeros(window_size),
        "mean": np.full(window_size, x[idx].mean()),
        "linear": np.linspace(left_end, right_end, window_size),
        "gaussian": np.random.randn(window_size),
        "gaussian_plus": x[idx] + np.random.randn(window_size),
    }[method]
    x[idx] = replace_values
    x = x.reshape(-1)
    return x
