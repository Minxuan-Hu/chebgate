import os
from chebgate.core.io import append_csv_row


def save_learning_curve_csv(logdir: str, row: dict):
    header = [
        "epoch",
        "train_loss",
        "train_acc",
        "true_train_loss",
        "true_train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "epoch_seconds",
        "wall_seconds",
    ]
    append_csv_row(os.path.join(logdir, "training_log.csv"), header, row)
