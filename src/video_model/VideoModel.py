import torch
from torch import nn
from torchvision.models import EfficientNet

from ..Config import Config


class VideoModel(nn.Module):
    def __init__(
        self,
        image_model: EfficientNet,
        n_classes: int,
        hidden_size: int = Config.gru_hidden_size,
    ):
        super().__init__()
        self.image_model = image_model
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=self.image_model.features[-1].out_channels,
            hidden_size=hidden_size,
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        outputs = []
        for sample in x:
            sample = self.image_model.features(sample)
            sample = nn.functional.adaptive_avg_pool2d(sample, (1, 1))
            sample = sample.squeeze(-1).squeeze(-1)
            _, sample = self.rnn(sample)
            sample = self.fc(sample)
            sample = self.softmax(sample)
            outputs.append(sample)
        return torch.concat(outputs)
