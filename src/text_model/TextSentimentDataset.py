from pathlib import Path

import numpy as np
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from ..Config import Config
from ..combinations.LabelsDataset import LabelsDataset


class TextSentimentDataset(LabelsDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_file_path: Path,
        undersample: bool = True,
    ):
        LabelsDataset.__init__(self, data_file_path, undersample)
        self.tokenizer = tokenizer

    def __getitem__(self, index: int):
        sentence, sentiment = self.sentences[index]
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
            Tensor(np.eye(len(self.sentiment_to_label))[label]),
        )
