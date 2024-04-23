from pathlib import Path

import numpy as np
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from Config import Config
from LabelsDataset import LabelsDataset


class TextSentimentDataset(LabelsDataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, data_file_path: Path):
        super().__init__(data_file_path)
        self.tokenizer = tokenizer

    def __getitem__(self, index: int):
        sentence, sentiment = self.sentences[index]
        label = self.sentiment_to_label[sentiment]
        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,
            max_length=Config.max_length,
            return_attention_mask=True,
            return_tensors="pt",
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
        )
        return (
            tokenized_sentence["input_ids"].squeeze(0),
            tokenized_sentence["attention_mask"].squeeze(0),
            Tensor(np.eye(len(self.sentiment_to_label))[label]),
        )
