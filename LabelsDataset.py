from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class LabelsDataset(Dataset):
    def __init__(self, data_file_path: Path):
        self.data_file_path = data_file_path
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
