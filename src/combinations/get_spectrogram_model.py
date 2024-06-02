import re

import torch

from ..Config import Config
from ..audio_spectrogram_model.AudioSpectrogramModel import AudioSpectrogramModel
from ..audio_spectrogram_model.train_audio_spectrogram import train_audio_spectrogram


def get_spectrogram_model() -> AudioSpectrogramModel:
    try:
        weights_path = max(
            Config.models_path.glob("audio_spectrogram_*.pth"),
            key=lambda path: int(re.findall(r"\d+", path.name)[-1]),
        )
        model = AudioSpectrogramModel(Config.n_classes).to(Config.device)
        model.load_state_dict(torch.load(weights_path))
    except ValueError:
        model = train_audio_spectrogram()
    return model
