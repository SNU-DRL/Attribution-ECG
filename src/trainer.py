# Reference: https://xylambda.github.io/blog/python/pytorch/machine-learning/2021/01/04/pytorch_trainer.html

import time

import pandas as pd
import torch
import torch.nn.functional as F
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

    def __init__(self, model, criterion, optimizer, scheduler, device, result_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.result_dir = result_dir

        self.model.to(self.device)
        self.metrics_df = pd.DataFrame(
            columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )

    def fit(self, train_loader, val_loader, epochs):
        """
        Parameters
        ----------
        train_loader :
        val_loader :
        epochs : int
            Number of training epochs.

        """
        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        for epoch in range(1, epochs + 1):
            # train
            train_loss, train_accuracy = self._train(train_loader)
            print(
                f"- [{epoch:03d}/{epochs:03d}] Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}, learning rate: {self.scheduler.get_last_lr()[0]:.6f}"
            )

            # validate
            val_loss, val_accuracy = self._validate(val_loader, is_test=False)
            print(
                f"- [{epoch:03d}/{epochs:03d}] Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}"
            )
            epoch_metrics_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_accuracy,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
            }
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

        probs = []
        labels = []

        for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
            x, y, _ = data_batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)

            self.optimizer.zero_grad()
            loss = self._compute_loss(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += y_hat.shape[0] * loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        total_loss /= len(loader.dataset)
        accuracy = self._compute_accuracy(labels, probs)

        return total_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss = 0.0

        probs = []
        labels = []

        for idx_batch, data_batch in enumerate(pbar := tqdm(loader)):
            x, y, _ = data_batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            _probs = F.softmax(y_hat, dim=1)

            loss = self._compute_loss(y_hat, y)
            total_loss += y_hat.shape[0] * loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            labels.extend(y)
            probs.extend(_probs)

        labels = torch.tensor(labels).detach().cpu().numpy()
        probs = torch.tensor(probs).detach().cpu().numpy()

        total_loss /= len(loader.dataset)
        accuracy = self._compute_accuracy(labels, probs)

        return total_loss, accuracy

    def _compute_loss(self, y_hat, y):
        loss = self.criterion(y_hat, y)

        # apply regularization if any
        # loss += penalty.item()

        return loss

    def _compute_accuracy(self, labels, probs):
        # y_train_pred_onehot = torch.argmax(y_train_pred, -1)
        # y_train_true_list.append(y_train_true)
        # y_train_pred_list.append(y_train_pred_onehot)

        # acc = (torch.cat(y_train_true_list) == torch.cat(y_train_pred_list)).float().mean()

        return 99.99
