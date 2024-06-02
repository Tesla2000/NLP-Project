from copy import deepcopy
from itertools import count

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from .BertAudioSpectrogramDataset import BertAudioSpectrogramDataset
from .BertAudioSpectrogramModel import BertAudioSpectrogramModel
from .get_bert_model import get_bert_model
from ..Config import Config


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for input_ids, attention_mask, spectrogram, labels in tqdm(loader):
        input_ids, attention_mask, spectrogram, labels = (
            input_ids.to(Config.device),
            attention_mask.to(Config.device),
            spectrogram.to(Config.device),
            labels.to(Config.device),
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, spectrogram)
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
    for input_ids, attention_mask, spectrogram, labels in tqdm(loader):
        input_ids, attention_mask, spectrogram, labels = (
            input_ids.to(Config.device),
            attention_mask.to(Config.device),
            spectrogram.to(Config.device),
            labels.to(Config.device),
        )
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, spectrogram)
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
    for input_ids, attention_mask, spectrogram, labels in tqdm(loader):
        input_ids, attention_mask, spectrogram, labels = (
            input_ids.to(Config.device),
            attention_mask.to(Config.device),
            spectrogram.to(Config.device),
            labels.to(Config.device),
        )
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, spectrogram)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Testing Accuracy: {accuracy:.2f}%")
    return accuracy


def train_bert_audio_spectrogram():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    bert_model = get_bert_model()
    num_classes = Config.n_classes
    model = BertAudioSpectrogramModel(bert_model, num_classes).to(Config.device)

    train_dataset = BertAudioSpectrogramDataset(
        tokenizer, Config.train_path, Config.train_spectograms_path
    )
    val_dataset = BertAudioSpectrogramDataset(
        tokenizer, Config.val_path, Config.val_spectograms_path
    )
    test_dataset = BertAudioSpectrogramDataset(
        tokenizer, Config.test_path, Config.test_spectograms_path
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
        Config.models_path.joinpath(f"text_audio_spectrogram_{accuracy:.4f}.pth"),
    )
