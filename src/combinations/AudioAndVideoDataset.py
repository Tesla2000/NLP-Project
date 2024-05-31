from pathlib import Path

import numpy as np
from torch import Tensor

from ..Config import Config
from ..audio_model.AudioDataset import AudioDataset
from ..audio_model.audio_preparation import (
    extract_audio_from_video,
    extract_audio_features,
)
from ..video_model.divide_video_to_frames import divide_video_to_frames


class AudioAndVideoDataset(AudioDataset):
    def __init__(
        self,
        data_file_path: Path,
        video_paths: Path,
    ):
        super().__init__(video_paths, data_file_path)

    def __getitem__(self, index: int):
        sentence, sentiment = self.sentences[index]
        divided_video = divide_video_to_frames(
            self.video_paths.joinpath(self.files[index]),
        )
        try:
            divided_video = divided_video.transpose((0, 3, 1, 2))
            audio_array, sampling_rate = extract_audio_from_video(
                self.video_paths.joinpath(self.files[index])
            )
            extracted_features = extract_audio_features(audio_array, sampling_rate)
        except OSError:
            return self.__getitem__(index - 1)

        label = self.sentiment_to_label[sentiment]
        return (
            Tensor(np.array(tuple(extracted_features.values()))).to(Config.device),
            Tensor(divided_video).to(Config.device),
            Tensor(np.eye(len(self.sentiment_to_label))[label]),
        )
