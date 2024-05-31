from pathlib import Path

import numpy as np
import torch

from ..Config import Config
from ..combinations.LabelsDataset import LabelsDataset
from ..video_model.divide_video_to_frames import divide_video_to_frames


class VideoDataset(LabelsDataset):
    def __init__(self, video_paths: Path, data_file_path: Path):
        super().__init__(data_file_path)
        self.video_paths = video_paths

    def __getitem__(self, index: int):
        sentence, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        divided_video = divide_video_to_frames(
            self.video_paths.joinpath(self.files[index])
        )
        try:
            divided_video = divided_video.transpose((0, 3, 1, 2))
        except OSError:
            return self.__getitem__(index - 1)
        return (
            torch.tensor(divided_video).to(Config.device).float(),
            torch.tensor(np.eye(len(self.sentiment_to_label))[label])
            .to(Config.device)
            .float(),
        )
