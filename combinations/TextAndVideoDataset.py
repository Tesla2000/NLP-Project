from pathlib import Path

import numpy as np
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from Config import Config
from text_model.TextSentimentDataset import TextSentimentDataset
from video_model.VideoDataset import VideoDataset
from video_model._divide_video_to_frames import _divide_video_to_frames


class TextAndVideoDataset(TextSentimentDataset, VideoDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_file_path: Path,
        video_paths: Path,
    ):
        super(TextAndVideoDataset, self).__init__(tokenizer, data_file_path)
        self.video_paths = video_paths

    def __getitem__(self, index: int):
        sentence, sentiment = self.sentences[index]
        # divided_video = _divide_video_to_frames(
        #     self.video_paths.joinpath(self.files[index]),
        # )
        # try:
        #     divided_video = divided_video.transpose((0, 3, 1, 2))
        # except:
        #     return self.__getitem__(
        #         index - 1
        #     )
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
            tokenized_sentence["attention_mask"].squeeze(0),
            # Tensor(divided_video).to(Config.device),
            Tensor(np.eye(len(self.sentiment_to_label))[label]),
        )
