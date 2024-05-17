from pathlib import Path
from audio_model.audio_preparation import extract_audio_from_video
from generate_spectrogram import generate_spectrogram

video_path = Path(
    r"datasets/MELD.eecs.umich.edu/MELD.Raw/train/train_splits/dia0_utt0.mp4"
)

audio_array, sampling_rate = extract_audio_from_video(video_path)
print("Audio Array Shape:", audio_array.shape)
print("Sampling Rate:", sampling_rate)
print("Data Type:", audio_array.dtype)

spectrogram = generate_spectrogram(audio_array, sampling_rate)
print("Spectrogram array:", spectrogram)
