from pathlib import Path
from typing import Generator
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from LabelsDataset import LabelsDataset
from audio_preparation import extract_audio_from_video, extract_audio_features

def _repeat_counter(length: int) -> Generator[int, None, None]:
    while True:
        yield from iter(
            tqdm(
                range(length),
                desc="Extracting audio features...",
                total=length,
                leave=False,
            )
        )

class AudioDataset(LabelsDataset):
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
            extracted_features = extract_audio_features(audio_array, sampling_rate)
        except OSError:
            print(f"Error processing {self.files[index]}, skipping.")
            return self.__getitem__(index - 1)

        return np.array(tuple(extracted_features.values())), label

def save_features_to_parquet(loader, output_file):
    all_features = []
    all_labels = []
    processed_videos = []

    for i, (features, label) in enumerate(loader):
        all_features.append(features.numpy().tolist())
        all_labels.append(label.numpy().tolist())
        processed_videos.append(loader.dataset.files[i])

        # Save features and labels to a parquet file after each iteration
        df = pd.DataFrame(all_features)
        df['label'] = all_labels
        df.columns = [str(col) for col in df.columns]  # Ensure all column names are strings
        print(f"Iteration {i+1}:")
        #print(df)
        df.to_parquet(output_file, index=False)

    # Save the list of processed videos
    with open(f"{output_file}_processed.txt", "w") as f:
        for video in processed_videos:
            f.write(f"{video}\n")

datasets_path = Path(r"datasets/MELD.eecs.umich.edu/MELD.Raw/")
train_video_path = datasets_path / "train/train_splits"
val_video_path = datasets_path / "dev/dev_splits_complete"
#val_video_path = datasets_path / "dev_1/"
test_video_path = datasets_path / "test/output_repeated_splits_test"
train_path = datasets_path / "train/train_sent_emo.csv"
val_path = datasets_path / "dev_sent_emo.csv"
test_path = datasets_path / "test_sent_emo.csv"

#val_dataset = AudioDataset(val_video_path, val_path)
train_dataset = AudioDataset(train_video_path, train_path)
#test_dataset = AudioDataset(test_video_path, test_path)

#val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#save_features_to_parquet(val_loader, "val_features.parquet")
#save_features_to_parquet(test_loader, "test_features.parquet")
save_features_to_parquet(train_loader, "train_features.parquet")

