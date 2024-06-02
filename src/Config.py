import random
from pathlib import Path

import numpy as np
import torch


class Config:
    learning_rate_spectrogram = 1e-5
    combined_hidden_size = 128
    val_video_length = 100
    n_classes = 3
    gru_hidden_size = 1024
    eps = 1e-8
    learning_rate = 2e-6
    consecutive_lacks_of_improvement_allowed = 2

    max_length = 64
    text_batch_size = 16
    video_batch_size = 1
    audio_batch_size = 16
    meld_dataset_path = Path("/kaggle/input/meld-dataset/MELD-RAW/MELD.Raw")
    features_dataset_path = Path("/kaggle/input/features-dataset")
    features_datasets_path = features_dataset_path / "features"
    train_video_path = meld_dataset_path / "train/train_splits"
    val_video_path = meld_dataset_path / "dev/dev_splits_complete"
    test_video_path = meld_dataset_path / "test/output_repeated_splits_test"
    train_path = meld_dataset_path / "train/train_sent_emo.csv"
    val_path = meld_dataset_path / "dev_sent_emo.csv"
    test_path = meld_dataset_path / "test_sent_emo.csv"
    train_features_path = features_datasets_path / "train_features_undersampled.parquet"
    val_features_path = features_datasets_path / "val_features_undersampled.parquet"
    test_features_path = features_datasets_path / "test_features_undersampled.parquet"
    train_text_features = features_dataset_path / "train_text_features.parquet"
    val_text_features = features_dataset_path / "val_text_features.parquet"
    test_text_features = features_dataset_path / "test_text_features.parquet"
    root = Path("/kaggle/working")
    temp_output_video_folder = root / "_temp_output_video_folder"
    models_path = root / "models"
    test_spectograms_path = root / "test_spectrograms"
    val_spectograms_path = root / "val_spectrograms"
    train_spectograms_path = root / "spectrograms"
    models_path.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
