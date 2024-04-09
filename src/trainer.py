# Reference: https://xylambda.github.io/blog/python/pytorch/machine-learning/2021/01/04/pytorch_trainer.html

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class Trainer:
    """
    Parameters
    ----------
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    scheduler : torch.optim.lr_scheduler
        Scheduler to perform the learning rate scheduler.
    result_dir : str
        Directory for saving results

    """

    def __init__(self, model, criterion, optimizer, scheduler, device, result_dir, num_classes, multi_label=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.result_dir = result_dir
        self.num_classes = num_classes
        self.multi_label = multi_label

        self.model.to(self.device)
        if multi_label:
            self.metric = "auc"
            self.metrics_df = pd.DataFrame(
                columns=["epoch", "train_loss", "train_auc", "val_loss", "val_auc"] + [f"auc_{class_idx}" for class_idx in range(self.num_classes)]
            )
        else:
            self.metric = "acc"
            self.metrics_df = pd.DataFrame(
                columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
            )

    def fit(self, train_loader, val_loader, epochs):
        """
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Dataloader for training set
        val_loader : torch.utils.data.DataLoader
            Dataloader for validation set
        epochs : int
            Number of training epochs.

        """
        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        for epoch in range(1, epochs + 1):
            # train
            if self.multi_label:
                train_loss, train_score, train_aucs = self._train(train_loader)
            else:
                train_loss, train_score = self._train(train_loader)
            print(
                f"- [{epoch:03d}/{epochs:03d}] Train loss: {train_loss:.4f}, Train {self.metric}: {train_score:.4f}, learning rate: {self.scheduler.get_last_lr()[0]:.6f}"
            )

            # validate
            if self.multi_label:
                val_loss, val_score, val_aucs = self._validate(val_loader)        
            else:
                val_loss, val_score = self._validate(val_loader)
            print(
                f"- [{epoch:03d}/{epochs:03d}] Validation loss: {val_loss:.4f}, Validation {self.metric}: {val_score:.4f}"
            )
            epoch_metrics_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                f"train_{self.metric}": train_score,
                "val_loss": val_loss,
                f"val_{self.metric}": val_score,
            }
            if self.multi_label:
                aucs_dict = {f"auc_{class_idx}": val_aucs[class_idx] for class_idx in range(self.num_classes)}
                epoch_metrics_dict.update(aucs_dict)
            self._log(epoch_metrics_dict)
            self.scheduler.step()
            print("*" * 30)

        total_time = time.time() - total_start_time

        # final message
        print(f"End of training. Total time: {round(total_time, 5)} seconds")
        self._save_model("last")

    def _save_model(self, desc):
        """
        - desc: model description (ex. 10epoch, best, last, ...)
        """
        torch.save(self.model, f"{self.result_dir}/model_{desc}.pt")

    def _log(self, epoch_metrics_dict):
        self.metrics_df = pd.concat(
            [self.metrics_df, pd.Series(epoch_metrics_dict).to_frame().T],
            ignore_index=True,
        )
        self.metrics_df = self.metrics_df.astype({"epoch": "int"})
        self.metrics_df.round(4).to_csv(f"{self.result_dir}/metrics.csv", index=False)

    def _train(self, loader):
        self.model.train()
        total_loss = 0.0

        labels = []
        probs = []

        for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
            _, x, y = data_batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            if self.multi_label:
                _probs = F.sigmoid(y_hat)
            else:
                _probs = F.softmax(y_hat, dim=1)

            self.optimizer.zero_grad()
            loss = self._compute_loss(y_hat, y)
            loss.backward()
            self.optimizer.step()
            total_loss += y_hat.shape[0] * loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            labels.extend(y)
            probs.extend(_probs)

        probs = torch.stack(probs).detach().cpu().numpy()
        total_loss /= len(loader.dataset)

        if self.multi_label:
            # multi-label
            labels = torch.stack(labels).detach().cpu().numpy()
            score, aucs = self._compute_auc_multi_label(labels, probs)
            return total_loss, score, aucs
        else:
            # single-label
            labels = torch.tensor(labels).detach().cpu().numpy()
            score = self._compute_accuracy(labels, probs)
            return total_loss, score


    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss = 0.0

        labels = []
        probs = []

        for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
            _, x, y = data_batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            if self.multi_label:
                _probs = F.sigmoid(y_hat)
            else:
                _probs = F.softmax(y_hat, dim=1)

            loss = self._compute_loss(y_hat, y)
            total_loss += y_hat.shape[0] * loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            labels.extend(y)
            probs.extend(_probs)

        probs = torch.stack(probs).detach().cpu().numpy()
        total_loss /= len(loader.dataset)

        if self.multi_label:
            # multi-label
            labels = torch.stack(labels).detach().cpu().numpy()
            score, aucs = self._compute_auc_multi_label(labels, probs)
            return total_loss, score, aucs
        else:
            # single-label
            labels = torch.tensor(labels).detach().cpu().numpy()
            score = self._compute_accuracy(labels, probs)
            return total_loss, score


    def _compute_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)

        # apply regularization if any
        # loss += penalty.item()

        return loss

    def _compute_accuracy(self, labels, probs):
        preds = np.argmax(probs, axis=1)
        acc = (labels == preds).mean()

        return acc

    def _compute_auc_multi_label(self, labels, probs):
        # One-vs-Rest multiclass ROC
        aucs = roc_auc_score(labels, probs, average=None, multi_class="ovr")
        macro_auc = np.mean(aucs)
        return macro_auc, aucs