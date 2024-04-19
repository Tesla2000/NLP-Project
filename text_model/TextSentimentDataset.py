from pathlib import Path

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from Config import Config


class TextSentimentDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, data_file_path: Path):
        self.data_file_path = data_file_path
        self.tokenizer = tokenizer
        df = pd.read_csv(data_file_path)
        sentiment_groups = df.groupby("Sentiment")
        self.sentences = tuple(
            (sentence, sentiment)
            for sentiment in sentiment_groups.indices
            for sentence in sentiment_groups.get_group(sentiment).values[:, 1]
        )
        self.sentiment_to_label = dict(
            map(reversed, enumerate(sentiment_groups.indices))
        )

    def __len__(self):
        return len(self.sentences)

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
