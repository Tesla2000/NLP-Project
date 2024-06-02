import torch
from torch import nn
from torchvision.models import resnet18
from transformers import BertForSequenceClassification


class BertAudioSpectrogramModel(nn.Module):
    n_text_features = 768

    def __init__(
        self,
        text_model: BertForSequenceClassification,
        n_classes: int,
    ):
        super().__init__()
        self.text_model = text_model

        self.resnet = resnet18()

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        num_ftrs = self.resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs, n_classes)
        self.res_net_weight = nn.Linear(1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        for param in self.text_model.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, spectrogram):
        text_model = self.text_model
        text_output = text_model(input_ids, attention_mask=attention_mask)
        for module in tuple(self.resnet._modules.values())[:-1]:
            spectrogram = module(spectrogram)
        spectrogram = torch.flatten(spectrogram, 1)
        spectrogram_output = self.res_net_weight.weight * self.softmax(
            self.fc(spectrogram)
        )
        return (text_output + spectrogram_output) / (1 + self.res_net_weight.weight)
