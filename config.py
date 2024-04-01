"""config.py"""

from pathlib import Path


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR.joinpath("data")
EMBEDDINGS_DIR = DATA_DIR.joinpath("embeddings")
MODELS_DIR = PROJECT_DIR.joinpath("models")
