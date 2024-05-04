from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from moviepy.editor import VideoFileClip
import opensmile


def extract_audio_from_video(video_path: Path) -> Tuple[np.ndarray, int]:
    video_clip = VideoFileClip(str(video_path))

    if video_clip.audio is None:
        video_clip.close()
        raise ValueError("No audio track in video")

    fps = 44100
    try:
        audio_frames = []
        for frame in video_clip.audio.iter_frames(fps=fps, dtype="float32"):
            audio_frames.append(frame)
        audio_array = np.vstack(audio_frames)
    finally:
        video_clip.close()

    return audio_array, fps


def extract_audio_features(audio: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    if audio.shape[1] > 1:
        audio = audio[:, 0]

    feature_entry = {}
    feature_level = opensmile.FeatureLevel.LowLevelDescriptors

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=feature_level,
        channels=[0],
    )
    features = smile.process_signal(audio, sampling_rate=sampling_rate)
    feature_names = smile.feature_names
    feature_means = features.mean(axis=0)

    for feature_name, mean_value in zip(feature_names, feature_means):
        feature_entry[f"{feature_name}"] = mean_value

    return feature_entry
