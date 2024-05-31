from typing import Callable, Sequence

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from ..Config import Config


def _train_model(
    feature_groups: Sequence[Tensor],
    labels: Tensor,
    model: nn.Module,
    loss_function: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
) -> None:
    model.train()
    feature_groups = tuple(group.to(Config.device) for group in feature_groups)
    labels = labels.to(Config.device)
    optimizer.zero_grad()
    logits = model(*feature_groups)
    train_loss = loss_function(logits, labels)
    train_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
