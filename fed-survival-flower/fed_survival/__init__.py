"""fed_survival: a federated Cox survival model for ALS, trained with Flower.

features.py    site-local: raw BRAINTEASER CSVs -> schema table -> encoded matrix
analytics.py   stage 1, federated analytics: builds the preprocessing artifact
cox.py         stage 2, federated Cox model (WebDISCO) and survival prediction
client_app.py  Flower ClientApp (a site)
server_app.py  Flower ServerApp (the coordinator)
"""

import tomllib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def read_config():
    """The run config ([tool.flwr.app.config] in pyproject.toml) for the stand-alone scripts.

    Relative data-path / output-dir are resolved against the project folder, so the scripts
    behave the same from any working directory.
    """
    with open(PROJECT_DIR / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)["tool"]["flwr"]["app"]["config"]
    for key in ("data-path", "output-dir"):
        config[key] = str(PROJECT_DIR / Path(config[key]).expanduser())
    return config
