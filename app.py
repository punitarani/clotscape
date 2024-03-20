"""app.py"""

import time

import cv2
import numpy as np
import onnxruntime
import torch
from segment_anything import sam_model_registry, SamPredictor

from config import EMBEDDINGS_DIR, MODELS_DIR

# Set fixed seeds for reproducibility across PyTorch and NumPy operations.
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)


# MedSAM model configs
SAM_MODEL_TYPE = "vit_h"
SAM_MODEL_PATH = MODELS_DIR.joinpath("vit", "sam_vit_h_4b8939.pth")
SAM_ONNX_PATH = MODELS_DIR.joinpath("vit", "sam_vit_h_int8.onnx")

if torch.backends.mps.is_available():
    device = torch.device("mps")  # available on macOS with Apple Silicon.
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading model...")
_tic = time.perf_counter()
SAM = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
ORT_SAM = onnxruntime.InferenceSession(SAM_ONNX_PATH)
print(f"Loaded model in {time.perf_counter() - _tic:0.2f}s")


IMAGE_NAME = "317947"
IMAGE_PATH = f"data/{IMAGE_NAME}.jpg"


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to a NumPy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: An image as a NumPy array with shape (height, width, channels).
    """
    image = cv2.imread(filename=image_path)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    print(f"Loaded image {image_path} with shape: {image.shape}")
    return image


def get_embedding(image: np.ndarray, name: str = None) -> np.ndarray:
    """
    Get the embedding for an image.
    Saves the embedding to disk if a name is provided.

    Args:
        image (np.ndarray): An image as a NumPy array with shape (height, width, channels).
        name (str): Name of the image to use for saving the embedding.

    Returns:
        np.ndarray: A 1D NumPy array representing the embedding for the image.
    """
    tic = time.perf_counter()

    # Load the embedding from disk if it exists
    if name:
        embedding_fp = EMBEDDINGS_DIR.joinpath(f"{name}.npy")
        if embedding_fp.exists():
            print(f"Loading embedding from {embedding_fp}")
            return np.load(file=embedding_fp)

    # Generate the embedding
    predictor = SamPredictor(sam_model=SAM)
    predictor.set_image(image)
    embedding = predictor.get_image_embedding().cpu().numpy()

    # Save the embedding to disk
    if name:
        np.save(file=EMBEDDINGS_DIR.joinpath(f"{name}.npy"), arr=embedding)

    print(f"Generated embedding in {time.perf_counter() - tic:0.2f}s")
    return embedding


if __name__ == "__main__":
    IMAGE = load_image(image_path=IMAGE_PATH)
    EMBEDDING = get_embedding(image=IMAGE, name=IMAGE_NAME)
    print(f"Embedding shape: {EMBEDDING.shape}")
