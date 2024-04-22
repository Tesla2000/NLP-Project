from torch import nn
from torchvision.models import EfficientNet

from Config import Config


class VideoModel(nn.Module):
    def __init__(
        self, image_model: EfficientNet, hidden_size: int = Config.gru_hidden_size
    ):
        super().__init__()
        self.image_model = image_model
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=self.image_model.features[-1].out_channels,
            hidden_size=hidden_size,
        )

    def forward(self, x):
        x = self.image_model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1).squeeze(-1)
        _, x = self.rnn(x)
