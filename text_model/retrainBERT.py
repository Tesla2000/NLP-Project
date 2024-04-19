from itertools import count
from math import ceil

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from Config import Config
from text_model.TextSentimentDataset import TextSentimentDataset


def retrainBert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_dataset = TextSentimentDataset(tokenizer, Config.train_csv)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = TextSentimentDataset(tokenizer, Config.eval_csv)
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
        lr=1e-3,
    )
    last_layer = nn.Softmax(dim=1)
    loss_function = nn.CrossEntropyLoss()
    total = ceil(len(train_dataset) / batch_size)
    loss = float("inf")
    for epoch in count(1):
        for input_ids, attention_mask, labels in tqdm(
            train_loader,
            total=total,
            desc=f"Previous loss {loss}. Training epoch {epoch}...",
        ):
            input_ids = input_ids.to(Config.device)
            attention_mask = attention_mask.to(Config.device)
            labels = labels.to(Config.device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask).logits
            outputs = last_layer(logits)
            loss = loss_function(labels, outputs)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            input_ids, attention_mask, labels = next(iter(eval_loader))
            input_ids = input_ids.to(Config.device)
            attention_mask = attention_mask.to(Config.device)
            labels = labels.to(Config.device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            outputs = last_layer(logits)
            loss = loss_function(labels, outputs)
            optimizer.step()
            loss = loss.item()


if __name__ == "__main__":
    retrainBert()
