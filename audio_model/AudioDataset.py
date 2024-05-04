from pathlib import Path

import numpy as np

from combinations.LabelsDataset import LabelsDataset
from .audio_preparation import (
    extract_audio_from_video,
    extract_audio_features,
)


class AudioDataset(LabelsDataset):
    def __init__(self, video_paths: Path, data_file_path: Path):
        LabelsDataset.__init__(self, data_file_path)
        self.video_paths = video_paths

    def __getitem__(self, index: int):
        _, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        try:
            audio_array, sampling_rate = extract_audio_from_video(
                self.video_paths.joinpath(self.files[index])
            )
            extracted_features = extract_audio_features(audio_array, sampling_rate)
        except OSError:
            return self.__getitem__(index - 1)

        return np.array(tuple(extracted_features.values())), label
