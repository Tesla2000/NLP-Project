from copy import deepcopy
from itertools import count
from math import ceil

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from tqdm import tqdm
from transformers import BertTokenizer

from ..Config import Config
from ..combinations.AudioAndTextAndVideoDataset import AudioAndTextAndVideoDataset
from ..combinations.AudioAndTextAndVideoModel import AudioAndTextAndVideoModel
from ..combinations._eval_model import _eval_model
from ..combinations._train_model import _train_model


def train_audio_and_text_and_video():
    dataset_type = AudioAndTextAndVideoDataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_dataset = dataset_type(tokenizer, Config.train_path, Config.train_video_path)
    batch_size = min(Config.text_batch_size, Config.audio_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = dataset_type(tokenizer, Config.val_path, Config.val_video_path)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    model = AudioAndTextAndVideoModel(
        image_model=efficientnet_b0(), n_classes=Config.n_classes
    )
    model.to(Config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        eps=Config.eps,
    )
    loss_function = nn.BCEWithLogitsLoss()
    total = ceil(len(train_dataset) / batch_size)
    best_accuracy = 0
    best_loss = float("inf")
    new_loss = float("inf")
    consecutive_lack_of_improvement = 0
    for epoch in count():
        for *feature_groups, labels in tqdm(
            train_loader,
            total=total,
            desc=f"Previous {accuracy=:.4f}, {new_loss=:.4f}. Training epoch {epoch}..."
            if best_accuracy
            else f"Training epoch {epoch}...",
        ):
            _train_model(feature_groups, labels, model, loss_function, optimizer)
        new_loss, accuracy = _eval_model(eval_loader, model, loss_function, total)
        if best_loss <= new_loss:
            consecutive_lack_of_improvement += 1
            if (
                Config.consecutive_lacks_of_improvement_allowed
                == consecutive_lack_of_improvement
            ):
                break
        else:
            best_loss = new_loss
            best_accuracy = accuracy
            best_model = deepcopy(model.state_dict())
            consecutive_lack_of_improvement = 0
    torch.save(best_model, Config.models_path.joinpath(f"{best_accuracy}.pth"))


if __name__ == "__main__":
    train_audio_and_text_and_video()
