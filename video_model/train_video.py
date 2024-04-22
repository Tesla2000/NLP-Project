from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from Config import Config
from video_model.VideoDataset import VideoDataset
from video_model.VideoModel import VideoModel


def train_video():
    model = VideoModel(image_model=efficientnet_b0())
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    train_dataset = VideoDataset(Config.train_video_path, Config.train_path)
    train_dataloader = DataLoader(train_dataset)
    for frames, label in train_dataloader:
        optimizer.zero_grad()
        output = model(frames)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train_video()
