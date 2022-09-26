import numpy as np
import torch
import torch.nn.functional as F

ds_beat_names = {
    0: 'undefined',     # Undefined
    1: 'normal',        # Normal
    2: 'pac',           # ESSV (PAC)
    3: 'aberrated',     # Aberrated
    4: 'pvc'            # ESV (PVC)
}

@torch.no_grad()
def evaluate_attr_x(x, model, attr_x, true_label, raw_label, min_prob=0, max_prob=1):
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
    
    boundaries_per_label = get_boundaries_per_label(raw_label)
    flat_raw_label = flatten_raw_label(raw_label)
    flat_raw_label = dict(sorted(flat_raw_label.items()))
    def localization_error(attr_x, true_label, boundaries_per_label, **kwargs):
        N = 0
        true_idx = []
        for boundary in boundaries_per_label[true_label]:
            N += boundary[1] - boundary[0]
            true_idx += list(np.arange(*boundary))
        attr_topN = np.argsort(attr_x)[-N:]
        true_idx = set(true_idx)
        pred_idx = set(attr_topN)

        iou = len(pred_idx & true_idx) / len(pred_idx | true_idx)
        acc = len(pred_idx - (pred_idx - true_idx)) / len(true_idx)

        return {'acc': acc, 'iou': iou}

    def pointing_game(attr_x, true_label, boundaries_per_label, **kwargs):
        attr_top1 = np.argmax(attr_x)
        
        correct = False
        for boundary in boundaries_per_label[true_label]:
            correct = correct or (attr_top1 in range(*boundary))
        return {'correct': correct}
    
    @torch.no_grad()
    def perturbation_based(attr_x, true_label, boundaries_per_label, x, model, min_prob=0, max_prob=1, replace='zero', window_size=16, **kwargs):
        """
        replace
        - zero: replace the erased part with 0
        - mean: replace the erased part with the mean of each edge
        - linear: replace the erased part with the linear interpolation of each edge
        """
        def perturb(x, idx, replace, window_size=window_size):
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
                'zero': np.zeros(window_size),
                'mean': np.full(window_size, (left_end + right_end) / 2),
                'linear': np.linspace(left_end, right_end, window_size)
            }[replace]
            x[idx] = replace_values
            x = x.reshape(-1)
            return x
        
        model = model.cuda()
        
        new_x, attr_x = np.copy(x[:-1]), attr_x[:-1]
        attr_x_reshaped = attr_x.reshape(-1, window_size)
        score_per_window = attr_x_reshaped.sum(1)
        LeRF_rank = np.argsort(score_per_window)
        MoRF_rank = LeRF_rank[::-1]

        LeRF_prob_list = []
        MoRF_prob_list = []

        init_prob = F.softmax(model(torch.tensor(new_x).reshape(1, 1, 1, -1).cuda()), dim=1)[0, true_label].item()
        LeRF_prob_list.append(init_prob)
        MoRF_prob_list.append(init_prob)
        
        new_x = np.copy(x[:-1])
        for window_idx in LeRF_rank:
            new_x = perturb(new_x, window_idx, replace=replace)
            LeRF_prob = F.softmax(model(torch.tensor(new_x).reshape(1, 1, 1, -1).cuda()), dim=1)[0, true_label].item()
            LeRF_prob_list.append(LeRF_prob)
        
        new_x = np.copy(x[:-1])
        for window_idx in MoRF_rank:
            new_x = perturb(new_x, window_idx, replace=replace)
            MoRF_prob = F.softmax(model(torch.tensor(new_x).reshape(1, 1, 1, -1).cuda()), dim=1)[0, true_label].item()
            MoRF_prob_list.append(MoRF_prob)

        normalized_LeRF_prob_list = (np.array(LeRF_prob_list) - min_prob) / (max_prob - min_prob)
        normalized_MoRF_prob_list = (np.array(MoRF_prob_list) - min_prob) / (max_prob - min_prob)

        area = (normalized_LeRF_prob_list - normalized_MoRF_prob_list).sum()
        return {'area': area, 'LeRF': normalized_LeRF_prob_list.tolist(), 'MoRF': normalized_MoRF_prob_list.tolist()}
    
    loc_metric = localization_error(attr_x, true_label, boundaries_per_label, x=x, model=model)
    pnt_metric = pointing_game(attr_x, true_label, boundaries_per_label, x=x, model=model)
    per_metric = perturbation_based(attr_x, true_label, boundaries_per_label, x=x, model=model, 
                                    replace='zero', min_prob=min_prob, max_prob=max_prob)
    return loc_metric, pnt_metric, per_metric


def get_boundaries_per_label(raw_label):
    flat_raw_label = flatten_raw_label(raw_label)
    flat_raw_label = dict(sorted(flat_raw_label.items()))

    r_peaks = np.array(list(flat_raw_label.keys()))
    beat_boundaries = (r_peaks[1:] + r_peaks[:-1]) // 2
    beat_boundaries = np.insert(beat_boundaries, 0, 0)
    beat_boundaries = np.append(beat_boundaries, 2048)
    beat_boundary_per_beat = [(s, e) for s, e in zip(beat_boundaries[:-1], beat_boundaries[1:])]
    beat_boundaries_per_label_dict = {
        0: [],
        1: [],
        2: []
    }
    
    for l, b in zip(flat_raw_label.values(), beat_boundary_per_beat):
        if l not in ['normal', 'pac', 'pvc']:
            continue
        beat_boundaries_per_label_dict[{'normal': 0, 'pac': 1, 'pvc': 2}[l]].append(b)
    
    return beat_boundaries_per_label_dict


def flatten_raw_label(raw_label):
    raw_label_dict = {}
    for i, idx in enumerate(raw_label):
        for j in idx:
            raw_label_dict[j] = ds_beat_names[i]
    return raw_label_dict