from copy import deepcopy
from itertools import count, islice
from math import ceil

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from tqdm import tqdm

from ..Config import Config
from ..video_model.VideoDataset import VideoDataset
from ..video_model.VideoModel import VideoModel


def train_video():
    batch_size = 1
    model = VideoModel(image_model=efficientnet_b0(), n_classes=Config.n_classes).to(
        Config.device
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    train_dataset = VideoDataset(Config.train_video_path, Config.train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total = ceil(len(train_dataset) / batch_size)
    best_accuracy = 0
    best_loss = float("inf")
    new_loss = float("inf")
    consecutive_lack_of_improvement = 0
    for epoch in count(1):
        model.train()
        for frames, labels in tqdm(
            train_dataloader,
            total=total,
            desc=f"Previous {accuracy=:.4f}, {new_loss=:.4f}. Watching friends for the {epoch} time..."
            if best_accuracy
            else f"Watching friends for the {epoch} time...",
        ):
            optimizer.zero_grad()
            outputs = model(frames)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        new_loss = 0.0
        label_ids = []
        predictions = []
        val_dataset = VideoDataset(Config.val_video_path, Config.val_path)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        for frames, labels in tqdm(
            islice(val_dataloader, Config.val_video_length),
            desc="Evaluating...",
            total=Config.val_video_length,
        ):
            label_ids += list(labels.to("cpu").numpy())
            with torch.no_grad():
                optimizer.zero_grad()
                outputs = model(frames)
                loss = loss_function(outputs, labels)
                predictions += list(outputs.cpu().numpy())
                new_loss += loss.item()
                optimizer.step()
        accuracy = accuracy_score(
            np.argmax(label_ids, axis=1), np.argmax(np.array(predictions), axis=1)
        )
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
        torch.save(
            best_model,
            Config.temp_output_video_folder.joinpath(f"video_{best_accuracy}.pth"),
        )


if __name__ == "__main__":
    train_video()
