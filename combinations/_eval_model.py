from typing import Callable, NamedTuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from Config import Config


class _ReturnedItem(NamedTuple):
    loss: float
    accuracy: float


@torch.no_grad()
def _eval_model(
    eval_loader: DataLoader,
    model: nn.Module,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    total: int,
) -> _ReturnedItem:
    model.eval()
    new_loss = 0
    predictions = []
    all_labels = []
    for input_ids, attention_mask, audio_features, labels in tqdm(
        eval_loader,
        total=total,
        desc="Evaluating...",
    ):
        input_ids = input_ids.to(Config.device)
        attention_mask = attention_mask.to(Config.device)
        labels = labels.to(Config.device)
        logits = model(audio_features, input_ids, attention_mask=attention_mask)
        new_loss = loss_function(logits, labels).item()
        predictions.append(logits.detach().cpu().numpy())
        all_labels.append(labels.to("cpu").numpy())
    accuracy = accuracy_score(
        np.argmax(np.concatenate(all_labels), axis=1),
        np.argmax(np.concatenate(predictions), axis=1),
    )
    return _ReturnedItem(new_loss, accuracy)
