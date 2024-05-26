import xgboost as xgb
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from Config import Config
from .AudioDataset import AudioDataset


def train_and_evaluate_xgboost():
    train_dataset = AudioDataset(Config.train_features_path, Config.train_path)
    val_dataset = AudioDataset(Config.val_features_path, Config.val_path)
    test_dataset = AudioDataset(Config.test_features_path, Config.test_path)
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_features, train_labels = next(iter(train_loader))

    dtrain = xgb.DMatrix(train_features, label=train_labels)
    params = {
        "objective": "multi:softmax",
        "num_class": Config.n_classes,
        "max_depth": 6,
        "eta": 0.3,
        "verbosity": 1,
    }
    num_boost_round = 100

    bst = xgb.train(params, dtrain, num_boost_round)

    val_features, val_labels = next(iter(val_loader))
    dval = xgb.DMatrix(val_features)
    val_preds = bst.predict(dval)
    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    test_features, test_labels = next(iter(test_loader))
    dtest = xgb.DMatrix(test_features)
    test_preds = bst.predict(dtest)
    test_accuracy = accuracy_score(test_labels, test_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    bst.save_model(Config.models_path.joinpath("bst.json"))
    return bst


if __name__ == "__main__":
    bst_model = train_and_evaluate_xgboost()
