import torch
from transformers import BertForSequenceClassification

from src.Config import Config
from src.text_model.retrainBERT import retrainBERT


def get_bert_model() -> BertForSequenceClassification:
    try:
        weights_path = next(Config.models_path.glob("text_*.pth"))
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=Config.n_classes,
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(Config.device)
        model.load_state_dict(torch.load(weights_path))
    except StopIteration:
        model = retrainBERT()
    return model
