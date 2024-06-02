import torch
from torch import nn
from transformers import BertForSequenceClassification

from ..Config import Config


class TextAndAudioModel(nn.Module):
    n_text_features = 768
    n_audio_features = 65

    def __init__(
        self,
        text_model: BertForSequenceClassification,
        n_classes: int,
    ):
        super().__init__()
        self.text_model = text_model
        self.fc = nn.Linear(
            in_features=self.n_text_features + self.n_audio_features,
            out_features=n_classes,
        )
        self.softmax = nn.Softmax(dim=1)
        for param in self.text_model.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, audios_features):
        text_model = self.text_model
        texts_features = text_model.bert(
            input_ids, attention_mask=attention_mask
        ).pooler_output
        sample = torch.concatenate(
            (audios_features.to(Config.device), texts_features), dim=1
        )
        sample = self.fc(sample.float())
        sample = self.softmax(sample)
        return sample
