from copy import deepcopy
from itertools import count
from math import ceil
from pathlib import Path

import torch
from keras.src.metrics import accuracy
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from tqdm import tqdm

from Config import Config
from video_model.VideoDataset import VideoDataset
from video_model.VideoModel import VideoModel


def train_video():
    model = VideoModel(image_model=efficientnet_b0(), n_classes=Config.n_classes).to(
        Config.device
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    train_dataset = VideoDataset(Config.train_video_path, Config.train_path)
    train_dataloader = DataLoader(train_dataset)
    val_dataset = VideoDataset(Config.val_video_path, Config.val_path)
    val_dataloader = DataLoader(val_dataset)
    batch_size = Config.batch_size
    total = ceil(len(train_dataset) / batch_size)
    best_accuracy = 0
    best_loss = float("inf")
    new_loss = float("inf")
    consecutive_lack_of_improvement = 0
    for epoch in count():
        model.train()
        for frames, label in tqdm(
            train_dataloader,
            total=total,
            desc=f"Previous {accuracy=:.4f}, {new_loss=:.4f}. Training epoch {epoch}..."
            if best_accuracy
            else f"Training epoch {epoch}...",
        ):
            optimizer.zero_grad()
            output = model(frames)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
        model.eval()
        new_loss = 0.0
        for frames, label in val_dataloader:
            with torch.no_grad():
                optimizer.zero_grad()
                output = model(frames)
                loss = loss_function(output, label)
                new_loss += loss.item()
                optimizer.step()
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
    out_folder = Path("models")
    out_folder.mkdir(exist_ok=True)
    torch.save(best_model, out_folder.joinpath(f"video_{best_accuracy}.pth"))


if __name__ == "__main__":
    train_video()
