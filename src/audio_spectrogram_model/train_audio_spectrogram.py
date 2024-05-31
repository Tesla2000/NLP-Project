import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from ..Config import Config
from AudioSpectrogramDataset import AudioSpectrogramDataset
from AudioSpectrogramModel import AudioSpectrogramModel


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
    return running_loss / len(loader)


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Testing Accuracy: {accuracy:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = ...
    model = AudioSpectrogramModel(num_classes=num_classes).to(device)

    train_dataset = AudioSpectrogramDataset(Config.train_video_path, Config.train_path)
    val_dataset = AudioSpectrogramDataset(Config.val_video_path, Config.val_path)
    test_dataset = AudioSpectrogramDataset(Config.test_video_path, Config.test_path)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = ...

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    test(model, test_loader, device)


if __name__ == "__main__":
    main()
