from pathlib import Path
from audio_preparation import extract_audio_from_video, extract_audio_features

video_path = Path(
    r"datasets/MELD.eecs.umich.edu/MELD.Raw/train/train_splits/dia0_utt0.mp4"
)

audio_array, sampling_rate = extract_audio_from_video(video_path)
print("Audio Array Shape:", audio_array.shape)
print("Sampling Rate:", sampling_rate)
print("Data Type:", audio_array.dtype)

features = extract_audio_features(audio_array, sampling_rate)
print("Extracted Features:", features)
