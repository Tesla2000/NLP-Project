from pathlib import Path

import pandas as pd

from ..combinations.LabelsDataset import LabelsDataset


class AudioDataset(LabelsDataset):
    def __init__(
        self, feature_path: Path, data_file_path: Path, undersample: bool = True
    ):
        LabelsDataset.__init__(self, data_file_path, undersample)
        self.feature_path = feature_path
        self.df = pd.read_parquet(feature_path).to_numpy()

    def __getitem__(self, index: int):
        _, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        return self.df[index], label
