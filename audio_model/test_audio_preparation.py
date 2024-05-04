from pathlib import Path
from _audio_preparation import _extract_audio_from_video, _extract_audio_features

video_path = Path(r'datasets/MELD.eecs.umich.edu/MELD.Raw/train/train_splits/dia0_utt0.mp4')

audio_array, sampling_rate = _extract_audio_from_video(video_path)
print("Audio Array Shape:", audio_array.shape)
print("Sampling Rate:", sampling_rate)
print("Data Type:", audio_array.dtype)

features = _extract_audio_features(audio_array, sampling_rate)
print("Extracted Features:", features)