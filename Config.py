import random
from pathlib import Path

import numpy as np
import torch


class Config:
    consecutive_lacks_of_improvement_allowed = 3
    val_path = None
    eps = 1e-8
    learning_rate = 2e-5
    n_epochs = 4
    max_length = 64
    datasets = {
        "LEGO": "http://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.125/research/DS/LEGO/LEGOv2.zip",
        "MELD": "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
    }
    root = Path(__file__).parent
    datasets_path = root / "datasets"
    train_path = datasets_path / "train.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
