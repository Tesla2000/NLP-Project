import re

from src.Config import Config
from download import download


def download_data():
    for dataset_name, dataset_link in Config.datasets.items():
        download(
            dataset_link,
            Config.meld_dataset_path.joinpath(
                f"{dataset_name}{re.findall(r'[^/][^.]*([^$]*)', dataset_link)[-1]}"
            ),
        )


if __name__ == "__main__":
    download_data()
