import copy
import numpy as np
from quantus.metrics.localisation.attribution_localisation import AttributionLocalisation
from quantus.metrics.localisation.auc import AUC
from quantus.metrics.localisation.pointing_game import PointingGame
from quantus.metrics.localisation.relevance_mass_accuracy import RelevanceMassAccuracy
from quantus.metrics.localisation.relevance_rank_accuracy import RelevanceRankAccuracy
from quantus.metrics.localisation.top_k_intersection import TopKIntersection
from quantus.metrics.faithfulness.region_perturbation import RegionPerturbation
from quantus.metrics.faithfulness.faithfulness_correlation import FaithfulnessCorrelation

from src.utils import aggregate_region_perturbation_scores, replace_by_zero


EVALUATION_METRICS = {
    "attribution_localization": AttributionLocalisation,
    "auc": AUC,
    "pointing_game": PointingGame,
    "relevance_mass_accuracy": RelevanceMassAccuracy,
    "relevance_rank_accuracy": RelevanceRankAccuracy,
    "top_k_intersection": TopKIntersection,
    "region_perturbation": RegionPerturbation,
    "faithfulness_correlation": FaithfulnessCorrelation
}


def evaluate_attribution(eval_metric, data_dict, attr_list, model, device, metric_kwargs):
    model.eval()
    updated_metric_kwargs = copy.deepcopy(metric_kwargs)
    if eval_metric in ["region_perturbation"]:
        updated_metric_kwargs.update({
            "perturb_func": replace_by_zero,
            "aggregate_func": aggregate_region_perturbation_scores,
            "return_aggregate": True
        })
    elif eval_metric in ["faithfulness_correlation"]:
        updated_metric_kwargs.update({
            "perturb_func": replace_by_zero,
            "return_aggregate": False
        })
        
    metric = EVALUATION_METRICS[eval_metric](display_progressbar=True, disable_warnings=True, **updated_metric_kwargs)

    x_batch = np.array(data_dict["x"])
    y_batch = np.array(data_dict["y"])
    a_batch = np.concatenate(attr_list)
    s_batch = np.array(list(map(build_segment_array, zip(data_dict["x"], data_dict["y"], data_dict["beat_spans"]))))

    if eval_metric in ["attribution_localization", "relevance_mass_accuracy"] and not metric_kwargs["abs"]:
        a_batch = np.clip(a_batch, 0, None)
    
    metric_scores = metric(model, x_batch, y_batch, a_batch, s_batch, channel_first=True, device=device)
    if len(metric_scores) == 1: # a nested list is returned when using custom aggregate_func
        metric_scores = metric_scores[0]
    
    return metric_scores
        
def build_segment_array(input_tuples):
    x, y, beat_spans = input_tuples
    s = np.zeros_like(x)
    for start, end in beat_spans[y]:
        s[:, :, start:end] = 1
    return s
