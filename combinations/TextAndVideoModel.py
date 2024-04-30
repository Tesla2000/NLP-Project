import torch
from torch import nn
from torchvision.models import EfficientNet
from transformers import BertForSequenceClassification

from Config import Config


class TextAndVideoModel(nn.Module):
    n_text_features = 768

    def __init__(
        self,
        image_model: EfficientNet,
        text_model: BertForSequenceClassification,
        n_classes: int,
        hidden_size: int = Config.gru_hidden_size,
    ):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=self.image_model.features[-1].out_channels,
            hidden_size=hidden_size,
        )
        self.fc = nn.Linear(
            in_features=hidden_size + self.n_text_features, out_features=n_classes
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, frames, input_ids, attention_mask):
        text_model = self.text_model
        texts_features = text_model.bert(
            input_ids, attention_mask=attention_mask
        ).pooler_output
        outputs = []
        for sample, text_features in zip(frames, texts_features):
            sample = self.image_model.features(sample)
            sample = nn.functional.adaptive_avg_pool2d(sample, (1, 1))
            sample = sample.squeeze(-1).squeeze(-1)
            _, sample = self.rnn(sample)
            sample = torch.concatenate((sample, text_features.unsqueeze(0)), dim=1)
            sample = self.fc(sample)
            sample = self.softmax(sample)
            outputs.append(sample)
        return torch.concat(outputs)
