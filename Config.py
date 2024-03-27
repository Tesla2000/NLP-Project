from pathlib import Path


class Config:
    datasets = {
        "LEGO": "http://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.125/research/DS/LEGO/LEGOv2.zip",
        "MELD": "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
    }
    root = Path(__file__).parent
    datasets_path = root / "datasets"
