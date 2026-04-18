import os
import yaml
import logging
import logging.config
from pathlib import Path


def setup_logging(config_path: str = "config/logging.yaml") -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Logging config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)