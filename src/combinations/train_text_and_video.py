from copy import deepcopy
from itertools import count
from math import ceil

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from ..Config import Config
from ..combinations.TextAndVideoDataset import TextAndVideoDataset
from ..combinations.TextAndVideoModel import TextAndVideoModel
from ..combinations._eval_model import _eval_model


def train_text_and_video():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_dataset = TextAndVideoDataset(
        tokenizer, Config.train_path, Config.train_video_path
    )
    batch_size = min(Config.text_batch_size, Config.video_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = TextAndVideoDataset(
        tokenizer, Config.val_path, Config.val_video_path
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    )
    model = TextAndVideoModel(
        image_model=efficientnet_b0(), text_model=bert_model, n_classes=Config.n_classes
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
        model.train()
        for input_ids, attention_mask, frames, labels in tqdm(
            train_loader,
            total=total,
            desc=f"Previous {accuracy=:.4f}, {new_loss=:.4f}. Training epoch {epoch}..."
            if best_accuracy
            else f"Training epoch {epoch}...",
        ):
            input_ids = input_ids.to(Config.device)
            attention_mask = attention_mask.to(Config.device)
            labels = labels.to(Config.device)
            optimizer.zero_grad()
            logits = model(frames, input_ids, attention_mask=attention_mask)
            train_loss = loss_function(logits, labels)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
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
    train_text_and_video()
