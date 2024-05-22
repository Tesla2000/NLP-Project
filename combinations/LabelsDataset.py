import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class LabelsDataset(Dataset):
    def __init__(self, data_file_path: Path, undersample: bool = True):
        random.seed(42)
        self.data_file_path = data_file_path
        df = pd.read_csv(data_file_path)
        sentiment_groups = df.groupby("Sentiment")
        divided_to_sentiment_sentences = dict(
            (
                sentiment,
                tuple(sentiment_groups.get_group(sentiment).values[:, [1, 5, 6]]),
            )
            for sentiment in sentiment_groups.indices
        )
        if undersample:
            max_samples = min(map(len, divided_to_sentiment_sentences.values()))
            for key, value in tuple(divided_to_sentiment_sentences.items()):
                percentage = max_samples / len(value)
                divided_to_sentiment_sentences[key] = tuple(
                    filter(lambda _: random.random() < percentage, value)
                )
        self.sentences = tuple(
            (sentence[0], sentiment)
            for sentiment, sentences in divided_to_sentiment_sentences.items()
            for sentence in sentences
        )
        self.files = tuple(
            f"dia{sentence[1]}_utt{sentence[2]}.mp4"
            for sentiment, sentences in divided_to_sentiment_sentences.items()
            for sentence in sentences
        )
        self.sentiment_to_label = dict(
            map(reversed, enumerate(sentiment_groups.indices))
        )

    def __len__(self):
        return len(self.sentences)


if __name__ == "__main__":
    LabelsDataset(
        Path("/home/tesla/Filip Studia/master/NLP/NLP-Project/datasets/train.csv")
    )
