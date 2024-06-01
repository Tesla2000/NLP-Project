import torch
import torch.nn as nn
from torchvision.models import resnet18


class AudioSpectrogramModel(nn.Module):
    def __init__(self, num_classes: int):
        super(AudioSpectrogramModel, self).__init__()

        self.resnet = resnet18()

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.resnet(x))
