import torch
from torch import nn
from transformers import BertForSequenceClassification

from Config import Config


class TextAndAudioModel(nn.Module):
    n_text_features = 768
    n_audio_features = 65

    def __init__(
        self,
        text_model: BertForSequenceClassification,
        n_classes: int,
        combine_hidden_layer_size: int = Config.combined_hidden_size,
    ):
        super().__init__()
        self.text_model = text_model
        self.hidden = nn.Linear(
            in_features=self.n_audio_features + self.n_text_features,
            out_features=combine_hidden_layer_size,
        )
        self.fc = nn.Linear(
            in_features=combine_hidden_layer_size, out_features=n_classes
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, audios_features):
        text_model = self.text_model
        texts_features = text_model.bert(
            input_ids, attention_mask=attention_mask
        ).pooler_output
        sample = torch.concatenate((audios_features, texts_features), dim=1)
        sample = self.hidden(sample.float())
        sample = self.fc(sample)
        sample = self.softmax(sample)
        return sample
