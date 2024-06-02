from copy import deepcopy
from itertools import count

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from ..Config import Config
from .TextAudioSpectrogramDataset import TextAudioSpectrogramDataset
from .TextAudioSpectrogramModel import TextAudioSpectrogramModel


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for spectrograms, text_features, labels in tqdm(loader):
        spectrograms, text_features, labels = (
            spectrograms.to(Config.device),
            text_features.to(Config.device),
            labels.to(Config.device),
        )

        optimizer.zero_grad()
        outputs = model(spectrograms, text_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for spectrograms, text_features, labels in tqdm(loader):
        spectrograms, text_features, labels = (
            spectrograms.to(Config.device),
            text_features.to(Config.device),
            labels.to(Config.device),
        )
        with torch.no_grad():
            outputs = model(spectrograms, text_features)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy, running_loss / len(loader)


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    for spectrograms, text_features, labels in tqdm(loader):
        spectrograms, text_features, labels = (
            spectrograms.to(Config.device),
            text_features.to(Config.device),
            labels.to(Config.device),
        )
        with torch.no_grad():
            outputs = model(spectrograms, text_features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Testing Accuracy: {accuracy:.2f}%")
    return accuracy


def train_text_audio_spectrogram():
    num_classes = Config.n_classes
    model = TextAudioSpectrogramModel(num_classes=num_classes).to(Config.device)

    train_dataset = TextAudioSpectrogramDataset(
        Config.train_spectograms_path, Config.train_path, Config.train_text_features
    )
    val_dataset = TextAudioSpectrogramDataset(
        Config.val_spectograms_path, Config.val_path, Config.val_text_features
    )
    test_dataset = TextAudioSpectrogramDataset(
        Config.test_spectograms_path, Config.test_path, Config.test_text_features
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate_spectrogram)

    best_val_lost = float("inf")
    no_improvement_iterations = 0
    for epoch in count(1):
        print(f"Epoch {epoch}")
        train_loss = train(model, train_loader, criterion, optimizer)
        accuracy, val_loss = validate(model, val_loader, criterion)
        print(
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
        )
        if val_loss < best_val_lost:
            best_val_lost = val_loss
            best_state = deepcopy(model.state_dict())
            no_improvement_iterations = 0
        else:
            no_improvement_iterations += 1
        if no_improvement_iterations > Config.consecutive_lacks_of_improvement_allowed:
            break

    model.load_state_dict(best_state)
    accuracy = test(model, test_loader)
    torch.save(
        best_state,
        Config.models_path.joinpath(f"audio_spectrogram_{accuracy:.4f}.pth"),
    )
