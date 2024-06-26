import torch
import torch.nn as nn
from torchvision.models import resnet18


class TextAudioSpectrogramModel(nn.Module):
    def __init__(self, num_classes: int):
        super(TextAudioSpectrogramModel, self).__init__()

        self.resnet = resnet18()

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        num_ftrs = self.resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs + 768, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        spectrogram: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        for module in tuple(self.resnet._modules.values())[:-1]:
            spectrogram = module(spectrogram)
        spectrogram = torch.flatten(spectrogram, 1)
        features = torch.concat((spectrogram, text_features), dim=1)
        return self.softmax(self.fc(features))
