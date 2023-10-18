import numpy as np
from quantus.metrics.localisation.attribution_localisation import AttributionLocalisation
from quantus.metrics.localisation.auc import AUC
from quantus.metrics.localisation.pointing_game import PointingGame
from quantus.metrics.localisation.relevance_mass_accuracy import RelevanceMassAccuracy
from quantus.metrics.localisation.relevance_rank_accuracy import RelevanceRankAccuracy
from quantus.metrics.localisation.top_k_intersection import TopKIntersection
from quantus.metrics.faithfulness.region_perturbation import RegionPerturbation

from src.utils import aggregate_func

EVALUATION_METRICS = {
    "attribution_localization": AttributionLocalisation,
    "auc": AUC,
    "pointing_game": PointingGame,
    "relevance_mass_accuracy": RelevanceMassAccuracy,
    "relevance_rank_accuracy": RelevanceRankAccuracy,
    "top_k_intersection": TopKIntersection,
    "region_perturbation": RegionPerturbation,
}


def evaluate_attribution(eval_metric, data_dict, attr_list, model, device, metric_kwargs):
    model.eval()
    if eval_metric in ["region_perturbation"]:
        metric = EVALUATION_METRICS[eval_metric](display_progressbar=True, disable_warnings=True, return_aggregate=True, aggregate_func=aggregate_func, **metric_kwargs)
    else: # localization metrics
        metric = EVALUATION_METRICS[eval_metric](display_progressbar=True, disable_warnings=True, **metric_kwargs)
    
    n = 10
    
    x_batch = np.array(data_dict["x"])[:n]
    y_batch = np.array(data_dict["y"])[:n]
    a_batch = np.concatenate(attr_list)[:n]
    s_batch = np.array(list(map(build_segment_array, zip(data_dict["x"], data_dict["y"], data_dict["beat_spans"]))))[:n]

    if eval_metric in ["attribution_localization", "relevance_mass_accuracy"] and not metric_kwargs["abs"]:
        a_batch = np.clip(a_batch, 0, None)
    
    metric_results = metric(model, x_batch, y_batch, a_batch, s_batch, channel_first=True, device=device)
    
    return np.nanmean(metric_results)
        
def build_segment_array(input_tuples):
    x, y, beat_spans = input_tuples
    s = np.zeros_like(x)
    for start, end in beat_spans[y]:
        s[:, :, start:end] = 1
    return s
