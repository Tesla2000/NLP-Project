from pathlib import Path

import xgboost as xgb
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from AudioDataset import AudioDataset

from Config import Config

def train_and_evaluate_xgboost(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_features, train_labels = [], []
    for features, label in train_loader:
        train_features.extend(features.numpy())
        train_labels.extend(label.numpy())

    dtrain = xgb.DMatrix(train_features, label=train_labels)
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.3,
        'verbosity': 1
    }
    num_boost_round = 100

    bst = xgb.train(params, dtrain, num_boost_round)

    val_features, val_labels = [], []
    for features, label in val_loader:
        val_features.extend(features.numpy())
        val_labels.extend(label.numpy())
    dval = xgb.DMatrix(val_features)
    val_preds = bst.predict(dval)
    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    test_features, test_labels = [], []
    for features, label in test_loader:
        test_features.extend(features.numpy())
        test_labels.extend(label.numpy())
    dtest = xgb.DMatrix(test_features)
    test_preds = bst.predict(dtest)
    test_accuracy = accuracy_score(test_labels, test_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return bst

if __name__ == "__main__":
    train_dataset = AudioDataset(Config.train_video_path, Config.train_path)
    val_dataset = AudioDataset(Config.val_video_path, Config.val_path)
    test_dataset = AudioDataset(Config.test_video_path, Config.test_path)

    bst_model = train_and_evaluate_xgboost(train_dataset, val_dataset, test_dataset)