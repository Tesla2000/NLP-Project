from pathlib import Path
from typing import Generator

from tqdm import tqdm

from ..combinations.LabelsDataset import LabelsDataset
from ..audio_model.audio_preparation import extract_audio_from_video
from .generate_spectrogram import generate_spectrogram


def _repeat_counter(length: int) -> Generator[int, None, None]:
    while True:
        yield from iter(
            tqdm(
                range(length),
                desc="Generating spectrogram...",
                total=length,
                leave=False,
            )
        )


class AudioSpectrogramDataset(LabelsDataset):
    def __init__(self, video_paths: Path, data_file_path: Path):
        LabelsDataset.__init__(self, data_file_path)
        self.video_paths = video_paths
        self.counter = _repeat_counter(len(self))

    def __getitem__(self, index: int):
        next(self.counter)
        _, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        try:
            audio_array, sampling_rate = extract_audio_from_video(
                self.video_paths.joinpath(self.files[index])
            )
            spectrogram = generate_spectrogram(audio_array, sampling_rate)
        except OSError:
            return self.__getitem__(index - 1)

        return spectrogram, label
