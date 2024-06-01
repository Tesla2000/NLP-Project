from pathlib import Path

import numpy as np
import pandas as pd

from ..combinations.LabelsDataset import LabelsDataset


class AudioSpectrogramDataset(LabelsDataset):
    def __init__(self, spectrograms_path: Path, data_file_path: Path):
        LabelsDataset.__init__(self, data_file_path)
        self.spectrograms_path = spectrograms_path

    def __getitem__(self, index: int):
        _, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        try:
            spectrogram_path = self.spectrograms_path.joinpath(self.files[index])
            spectrogram = np.array(
                [pd.read_parquet(spectrogram_path.with_suffix(".parquet")).to_numpy()]
            )
        except OSError as e:
            print(e)
            return self.__getitem__(index - 1)

        return spectrogram, label
