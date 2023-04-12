import argparse
import os

import torch

from src.dataset import ECG_DataModule
from src.models.model_wrapper import ModelWrapper
from src.setup import setup
from src.trainer import Trainer


def main(args):
    # device
    device = setup(args)

    # dataloader
    data_module = ECG_DataModule(
        args.dataset_path, batch_size=args.batch_size, seed=args.seed
    )
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    # model
    model = ModelWrapper(args.model, num_classes=3)

    # hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1
    )

    trainer = Trainer(model, criterion, optimizer, scheduler, device, args.result_dir)
    trainer.fit(train_loader, test_loader, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attribution ECG")

    # Dataset
    parser.add_argument(
        "--dataset_path", default="dataset/12000_btype_new.pkl", type=str
    )

    # Model
    parser.add_argument("--model", default="resnet18_7", type=str)

    # Hyperparameters
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--weight_decay", default=1e-7, type=float)

    # Settings
    parser.add_argument("--gpu_num", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    # Result
    parser.add_argument("--result_dir", default="./result", type=str)

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    main(args)
