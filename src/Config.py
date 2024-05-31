import random
from pathlib import Path

import numpy as np
import torch


class Config:
    combined_hidden_size = 128
    val_video_length = 100
    n_classes = 3
    gru_hidden_size = 1024
    eps = 1e-8
    learning_rate = 2e-3
    consecutive_lacks_of_improvement_allowed = 3

    max_length = 64
    text_batch_size = 16
    video_batch_size = 1
    audio_batch_size = 16
    meld_dataset_path = Path("../datasets/meld")
    features_datasets_path = Path("../datasets/features")
    train_video_path = meld_dataset_path / "train/train_splits"
    val_video_path = meld_dataset_path / "dev/dev_splits_complete"
    test_video_path = meld_dataset_path / "test/output_repeated_splits_test"
    train_path = meld_dataset_path / "train/train_sent_emo.csv"
    val_path = meld_dataset_path / "dev_sent_emo.csv"
    test_path = meld_dataset_path / "test_sent_emo.csv"
    train_features_path = features_datasets_path / "train_features_undersampled.parquet"
    val_features_path = features_datasets_path / "val_features_undersampled.parquet"
    test_features_path = features_datasets_path / "test_features_undersampled.parquet"
    root = Path("..")
    temp_output_video_folder = root / "_temp_output_video_folder"
    models_path = root / "models"
    models_path.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
