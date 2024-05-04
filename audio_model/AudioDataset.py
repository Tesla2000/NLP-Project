from pathlib import Path

from combinations.LabelsDataset import LabelsDataset
from audio_model._audio_preparation import (
    _extract_audio_from_video,
    _extract_audio_features,
)


class AudioDataset(LabelsDataset):
    def __init__(self, video_paths: Path, data_file_path: Path):
        LabelsDataset.__init__(self, data_file_path)
        self.video_paths = video_paths

    def __getitem__(self, index: int):
        _, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        audio_array, sampling_rate = _extract_audio_from_video(
            self.video_paths.joinpath(self.files[index])
        )
        extracted_features = _extract_audio_features(audio_array, sampling_rate)

        return extracted_features, label
