import torch
from torch import nn
from torchvision.models import EfficientNet

from Config import Config


class AudioAndVideoModel(nn.Module):
    n_audio_features = 65

    def __init__(
        self,
        image_model: EfficientNet,
        n_classes: int,
        hidden_size: int = Config.gru_hidden_size,
        combine_hidden_layer_size: int = Config.combined_hidden_size,
    ):
        super().__init__()
        self.image_model = image_model
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=self.image_model.features[-1].out_channels,
            hidden_size=hidden_size,
        )
        self.hidden = nn.Linear(
            in_features=self.n_audio_features + hidden_size,
            out_features=combine_hidden_layer_size,
        )
        self.fc = nn.Linear(
            in_features=combine_hidden_layer_size, out_features=n_classes
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, audios_features, frames):
        outputs = []
        for sample, audio_features in zip(frames, audios_features):
            sample = self.image_model.features(sample)
            sample = nn.functional.adaptive_avg_pool2d(sample, (1, 1))
            sample = sample.squeeze(-1).squeeze(-1)
            _, sample = self.rnn(sample)
            sample = torch.concatenate(
                (sample.unsqueeze(0), audio_features.unsqueeze(0)), dim=1
            )
            outputs.append(sample)
        sample = torch.concat(outputs)
        sample = self.hidden(sample)
        sample = self.fc(sample)
        return self.softmax(sample)
