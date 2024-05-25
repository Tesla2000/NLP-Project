from pathlib import Path
from typing import Generator
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from LabelsDataset import LabelsDataset
from audio_preparation import extract_audio_from_video
import librosa
import cv2

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

def generate_spectrogram(audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, n_mels=128, fmax=8000)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB -= S_DB.min()
    S_DB /= S_DB.max()
    S_resized = cv2.resize(S_DB, (224, 224), interpolation=cv2.INTER_CUBIC)
    return S_resized

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
            print(f"Error processing {self.files[index]}, skipping.")
            return self.__getitem__(index - 1)

        return spectrogram, label

def save_spectrograms_to_parquet(loader, output_file):
    all_data = []
    processed_videos = []

    for i, (spectrogram, label) in enumerate(loader):
        all_data.append({'spectrogram': spectrogram, 'label': label})
        processed_videos.append(loader.dataset.files[i])

        df = pd.DataFrame(all_data)
        df['spectrogram'] = df['spectrogram'].apply(lambda x: x.tolist())

        df['label'] = df['label'].astype(int)
        df.to_parquet(output_file, index=False)

    with open(f"{output_file}_processed.txt", "w") as f:
        for video in processed_videos:
            f.write(f"{video}\n")

datasets_path = Path(r"datasets/MELD.eecs.umich.edu/MELD.Raw/")
train_video_path = datasets_path / "train/train_splits"
val_video_path = datasets_path / "dev/dev_splits_complete"
test_video_path = datasets_path / "test/output_repeated_splits_test"
train_path = datasets_path / "train/train_sent_emo.csv"
val_path = datasets_path / "dev_sent_emo.csv"
test_path = datasets_path / "test_sent_emo.csv"

# train_dataset = AudioSpectrogramDataset(train_video_path, train_path)
# val_dataset = AudioSpectrogramDataset(val_video_path, val_path)
test_dataset = AudioSpectrogramDataset(test_video_path, test_path)

# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# save_spectrograms_to_parquet(train_loader, "train_spectrograms.parquet")
# save_spectrograms_to_parquet(val_loader, "val_spectrograms.parquet")
save_spectrograms_to_parquet(test_loader, "test_spectrograms.parquet")