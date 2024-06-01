from copy import deepcopy
from itertools import count
from math import ceil

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from ..Config import Config
from .TextSentimentDataset import TextSentimentDataset


def retrainBERT():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_dataset = TextSentimentDataset(tokenizer, Config.train_path)
    batch_size = Config.text_batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = TextSentimentDataset(tokenizer, Config.val_path)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
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
        for input_ids, attention_mask, labels in tqdm(
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
            logits = model(input_ids, attention_mask=attention_mask)[0]
            train_loss = loss_function(logits, labels)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            input_ids, attention_mask, labels = next(iter(eval_loader))
            input_ids = input_ids.to(Config.device)
            attention_mask = attention_mask.to(Config.device)
            labels = labels.to(Config.device)
            logits = model(input_ids, attention_mask=attention_mask)[0]
            new_loss = loss_function(logits, labels).item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()
            accuracy = accuracy_score(
                np.argmax(label_ids, axis=1), np.argmax(logits, axis=1)
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
    torch.save(best_model, Config.models_path.joinpath(f"{best_accuracy}.pth"))
    model.load_state_dict(best_model)
    return model


if __name__ == "__main__":
    retrainBERT()
