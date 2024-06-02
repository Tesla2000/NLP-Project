from pathlib import Path

import numpy as np
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from ..audio_spectrogram_model.AudioSpectrogramDataset import AudioSpectrogramDataset
from ..text_model.TextSentimentDataset import TextSentimentDataset


class BertAudioSpectrogramDataset(TextSentimentDataset, AudioSpectrogramDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_file_path: Path,
        spectograms_path: Path,
        undersample: bool = True,
    ):
        TextSentimentDataset.__init__(
            self, tokenizer, text_file_path, undersample=undersample
        )
        AudioSpectrogramDataset.__init__(self, spectograms_path, text_file_path)

    def __getitem__(self, index: int):
        try:
            spectrogram, label = AudioSpectrogramDataset.__getitem__(
                self, index, throw=True
            )
        except OSError:
            return self.__getitem__(index - 1)
        input_ids, attention_mask, _ = TextSentimentDataset.__getitem__(self, index)
        return (
            input_ids.squeeze(0),
            attention_mask.squeeze(0),
            spectrogram,
            label,
        )
