import random
from pathlib import Path

import numpy as np
import torch


class Config:
    video_batch_size = 1
    n_classes = 3
    gru_hidden_size = 1024
    eps = 1e-8
    learning_rate = 2e-6
    consecutive_lacks_of_improvement_allowed = 3

    max_length = 64
    text_batch_size = 16
    datasets_path = Path("/kaggle/input/meld-dataset/MELD-RAW/MELD.Raw")
    train_video_path = datasets_path / "train/train_splits"
    val_video_path = datasets_path / "dev/dev_splits_complete"
    train_path = datasets_path / "train/train_sent_emo.csv"
    val_path = datasets_path / "dev_sent_emo.csv"
    root = Path(".")
    temp_output_video_folder = root / "_temp_output_video_folder"
    models_path = root / "models"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


for field_name in dir(Config):
    field = getattr(Config, field_name)
    if isinstance(field, Path) and not field.is_file():
        field.mkdir(parents=True, exist_ok=True)
