from pathlib import Path

import numpy as np
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from Config import Config
from audio_model.AudioDataset import AudioDataset
from audio_model.audio_preparation import (
    extract_audio_from_video,
    extract_audio_features,
)
from text_model.TextSentimentDataset import TextSentimentDataset


class TextAndAudioDataset(TextSentimentDataset, AudioDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_file_path: Path,
        video_paths: Path,
    ):
        super().__init__(tokenizer, data_file_path)
        self.video_paths = video_paths

    def __getitem__(self, index: int):
        sentence, sentiment = self.sentences[index]
        try:
            audio_array, sampling_rate = extract_audio_from_video(
                self.video_paths.joinpath(self.files[index])
            )
            extracted_features = extract_audio_features(audio_array, sampling_rate)
        except OSError:
            return self.__getitem__(index - 1)

        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,
            max_length=Config.max_length,
            return_attention_mask=True,
            return_tensors="pt",
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
        )
        label = self.sentiment_to_label[sentiment]
        return (
            tokenized_sentence["input_ids"].squeeze(0),
            tokenized_sentence["attention_mask"].squeeze(0),
            Tensor(np.array(tuple(extracted_features.values()))).to(Config.device),
            Tensor(np.eye(len(self.sentiment_to_label))[label]),
        )
