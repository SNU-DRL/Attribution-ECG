import torch
import numpy as np
import tqdm
import torch.nn.functional as F


class Evaluator:
    def __init__(self, model, loader, prob_threshold, device):
        self.model = model
        self.loader = loader
        self.prob_threshold = prob_threshold
        self.device = device

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def filter_dataset(self):
        for idx_batch, data_batch in enumerate(pbar := tqdm(self)):
            x, y, y_raw = data_batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            _probs = F.softmax(y_hat, dim=1)

        # normal 제외 (label 0)
        # label 1, label 2 중에서 해당 Prob이 threshold 이상인 sample들만 모아서
        # return X, y, y_raw, prob

        X, labels = pickle.load(gzip.GzipFile(args.dataset_path, "rb"))
        X = preprocess(X)
        X = np.expand_dims(X, [1, 2])
        y = np.array([l["btype"] for l in labels])
        y_raw = np.array([l["btype_raw"] for l in labels], dtype=object)

        """
        Load dataset
        """
        _, X_test_ds, _, y_test_ds, _, y_raw_test_ds = train_test_split(
            X, y, y_raw, train_size=6000, test_size=6000, stratify=y, random_state=seed
        )

        """
        Remove normal
        """
        class_idx = list(filter(lambda i: y_test_ds[i] != 0, np.arange(len(y_test_ds))))
        X_wo_normal = X_test_ds[class_idx]
        y_wo_normal = y_test_ds[class_idx]
        y_raw_wo_normal = y_raw_test_ds[class_idx]

        """
        Get probability
        """
        bs = 1024
        num_batch = len(class_idx) // bs

        X_above_thres = []
        y_above_thres = []
        y_raw_above_thres = []
        prob_above_thres = []

        for bn in range(num_batch + 1):
            X_batch = torch.from_numpy(X_wo_normal[bn * bs : (bn + 1) * bs]).cuda()
            y_batch = y_wo_normal[bn * bs : (bn + 1) * bs]
            y_raw_batch = y_raw_wo_normal[bn * bs : (bn + 1) * bs]

            y_softmax = to_np(F.softmax(model(X_batch), dim=1))
            y_prob = y_softmax[np.arange(len(y_softmax)), y_batch]

            is_above_thres = y_prob > prob_thres
            idx_filter = np.arange(len(y_prob))[is_above_thres]

            X_above_thres.append(to_np(X_batch)[idx_filter])
            y_above_thres.append(y_batch[idx_filter])
            y_raw_above_thres.append(y_raw_batch[idx_filter])
            prob_above_thres.append(y_prob[idx_filter])

        X_above_thres = np.concatenate(X_above_thres)
        y_above_thres = np.concatenate(y_above_thres)
        y_raw_above_thres = np.concatenate(y_raw_above_thres)
        prob_above_thres = np.concatenate(prob_above_thres)

        return X_above_thres, y_above_thres, y_raw_above_thres, prob_above_thres
