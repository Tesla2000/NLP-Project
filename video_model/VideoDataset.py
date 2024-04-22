from pathlib import Path

import numpy as np
import torch

from Config import Config
from LabelsDataset import LabelsDataset
from video_model._divide_video_to_frames import _divide_video_to_frames


class VideoDataset(LabelsDataset):
    def __init__(self, video_paths: Path, data_file_path: Path):
        super().__init__(data_file_path)
        self.video_paths = video_paths

    def __len__(self):
        return len(tuple(self.video_paths.iterdir()))

    def __getitem__(self, index: int):
        sentence, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        divided_video = _divide_video_to_frames(
            tuple(self.video_paths.iterdir())[index]
        )
        divided_video = divided_video.transpose((0, 3, 1, 2))
        return (
            torch.tensor(divided_video).to(Config.device).float(),
            torch.tensor(np.eye(len(self.sentiment_to_label))[label])
            .to(Config.device)
            .float(),
        )
