import torch
from torch import nn
from transformers import BertForSequenceClassification

from Config import Config


class TextAndAudioModel(nn.Module):
    n_text_features = 768
    n_audio_features = 75

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

    def forward(self, audios_features, input_ids, attention_mask):
        text_model = self.text_model
        texts_features = text_model.bert(
            input_ids, attention_mask=attention_mask
        ).pooler_output
        outputs = []
        for audio_features, text_features in zip(audios_features, texts_features):
            print(audios_features.shape, text_features.shape)
            sample = torch.concatenate(
                (audio_features, text_features.unsqueeze(0)), dim=1
            )
            sample = self.hidden(sample)
            sample = self.fc(sample)
            sample = self.softmax(sample)
            outputs.append(sample)
        return torch.concat(outputs)
