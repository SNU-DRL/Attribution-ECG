from quantus.metrics.localisation.attribution_localisation import AttributionLocalisation
from quantus.metrics.localisation.auc import AUC
from quantus.metrics.localisation.focus import Focus
from quantus.metrics.localisation.pointing_game import PointingGame
from quantus.metrics.localisation.relevance_mass_accuracy import RelevanceMassAccuracy
from quantus.metrics.localisation.relevance_rank_accuracy import RelevanceRankAccuracy
from quantus.metrics.localisation.top_k_intersection import TopKIntersection

import numpy as np

EVALUATION_METRICS = {
    "attribution_localization": AttributionLocalisation,
    "auc": AUC,
    "focus": Focus,
    "pointing_game": PointingGame,
    "relevance_mass_accuracy": RelevanceMassAccuracy,
    "relevance_rank_accuracy": RelevanceRankAccuracy,
    "top_k_intersection": TopKIntersection
}

def evaluate_attribution(eval_metric, data_dict, attr_list, model, device, absolute):
    model.eval()
    metric = EVALUATION_METRICS[eval_metric](abs=absolute, display_progressbar=True)

    x_batch = np.array(data_dict["x"])
    y_batch = np.array(data_dict["y"])
    a_batch = np.concatenate(attr_list)
    s_batch = np.array(list(map(build_segment_array, zip(data_dict["x"], data_dict["y"], data_dict["beat_spans"]))))

    metric_results = metric(model, x_batch, y_batch, a_batch, s_batch, channel_first=True, device=device)

    return np.mean(metric_results)
        
def build_segment_array(input_tuples):
    x, y, beat_spans = input_tuples
    s = np.zeros_like(x)
    for start, end in beat_spans[y]:
        s[:, :, start:end] = 1
    return s
