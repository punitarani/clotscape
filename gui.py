"""gui.py"""

import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
from PIL import Image
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
from skimage import io, transform
from torch.nn import functional as F

from config import MODELS_DIR

# Set fixed seeds for reproducibility across PyTorch and NumPy operations.
torch.manual_seed(2023)  # Seed for generating random numbers in pytorch
torch.cuda.empty_cache()  # Clears the CUDA memory cache to free unused pytorch memory
torch.cuda.manual_seed(2023)  # Seed for generating random numbers in CUDA
np.random.seed(2023)  # Seed for generating random numbers in numpy


# MedSAM model configs
SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = MODELS_DIR.joinpath("medsam", "medsam_vit_b.pth")
ONNX_MODEL_PATH = MODELS_DIR.joinpath("medsam", "medsam_vit_b_int8.onnx")
MEDSAM_IMG_INPUT_SIZE = 1024

if torch.backends.mps.is_available():
    # Use Metal Performance Shaders (MPS) if available on macOS with Apple Silicon.
    device = torch.device("mps")
else:
    # Use CUDA if available, otherwise use CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading MedSAM model...")
tic = time.perf_counter()

# set up model
MEDSAM_MODEL = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
MEDSAM_MODEL.eval()

MEDSAM_ONNX_MODEL = SamOnnxModel(MEDSAM_MODEL, return_single_mask=False)

print(f"Loaded MedSAM in {time.perf_counter() - tic:0.2f}s")


ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
    (192, 192, 192),
    (64, 64, 64),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 0, 127),
    (192, 0, 192),
]


@torch.no_grad()
def medsam_inference(
    medsam_model: torch.nn.Module,
    img_embed: torch.Tensor,
    box_1024: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Perform segmentation inference using the MedSAM model.

    Args:
        medsam_model (torch.nn.Module): The MedSAM model for segmentation.
        img_embed (torch.Tensor): The precomputed image embeddings from the model.
        box_1024 (np.ndarray): Normalized bounding box coordinates for the area of interest.
        height (int): The height of the original image.
        width (int): The width of the original image.

    Returns:
    - np.ndarray: The segmentation mask as a binary numpy array.
    """

    # Convert bounding box to a PyTorch tensor and adjust its shape.
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        # Adjust shape for model input.
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    # Generate embeddings for the specified bounding box.
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    # Decode the embeddings to generate segmentation logits.
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    # Apply sigmoid to logits and resize to original image dimensions.
    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)

    # Threshold the prediction to obtain a binary segmentation mask.
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def np2pixmap(np_img: np.ndarray) -> QPixmap:
    """
    Converts a NumPy array representing an image to a QPixmap for display in PyQt.

    Args:
        np_img (np.ndarray): An image as a NumPy array with shape (height, width, channels).

    Returns:
        QPixmap: A QPixmap object for display in PyQt GUI applications.
    """
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qImg)


class Window(QWidget):
    """PyQt application window for interactive segmentation."""

    def __init__(self):
        super().__init__()

        self.half_point_size = 5  # Radius of bbox starting and ending points

        # Initialize UI components and state variables
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None

        # Set up the graphics view and layout
        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)

        # Create and add buttons to the layout
        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")
        hbox = QHBoxLayout()
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        # Set up keyboard shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        # Class variables
        self.mask_c = None

        # Connect buttons to their actions
        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)

    def undo(self):
        """
        Reverts the last segmentation action by restoring the previous mask.
        """

        if self.prev_mask is None:
            print("No previous mask record")
            return

        # Decrement the color index to revert to the previous state.
        self.color_idx -= 1

        # Blend the previous mask with the original image for visual representation.
        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.prev_mask.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        # Update the GUI to display the reverted state.
        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        # Update the application state to reflect the undo action.
        self.mask_c = self.prev_mask
        self.prev_mask = None

    def load_image(self):
        """
        Loads an image from file and displays it in the application for segmentation.
        """
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        # Exit if no image file is selected.
        if file_path is None or len(file_path) == 0:
            print("No image path specified, please select an image")
            exit()

        # Read the image and convert to 3-channel RGB if necessary.
        img_np = io.imread(file_path)
        img_3c = (
            np.repeat(img_np[:, :, None], 3, axis=-1)
            if len(img_np.shape) == 2
            else img_np
        )

        # Store the image and its path, and calculate embeddings for segmentation.
        self.img_3c = img_3c
        self.image_path = file_path
        self.get_embeddings()

        # Display the image in the GUI.
        pixmap = np2pixmap(self.img_3c)
        self.scene = QGraphicsScene(0, 0, *self.img_3c.shape[:2])
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # Setup event handlers for interactive segmentation.
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, ev):
        """
        Handles mouse press events to start drawing a bounding box.
        """
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_move(self, ev):
        """
        Updates the drawing as the mouse moves with the button pressed.
        """
        if not self.is_mouse_down:
            return

        x, y = ev.scenePos().x(), ev.scenePos().y()

        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

        if self.rect is not None:
            self.scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
        )

    def mouse_release(self, ev):
        """
        Finalizes the bounding box and performs segmentation on the selected region.
        """
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        self.is_mouse_down = False

        H, W, _ = self.img_3c.shape
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        print("Bounding box:", box_np)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        sam_mask = medsam_inference(MEDSAM_MODEL, self.embedding, box_1024, H, W)

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sam_mask != 0] = COLORS[self.color_idx % len(COLORS)]
        self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

    def save_mask(self):
        """
        Saves the current mask to a file.
        """
        out_path = f"{self.image_path.split('.')[0]}_mask.png"
        io.imsave(out_path, self.mask_c)

    @torch.no_grad()
    def get_embeddings(self):
        """
        Computes and stores the image embeddings using the MedSAM model.

        This method preprocesses the loaded image by resizing it to 1024x1024 pixels,
        normalizing its pixel values, and then computes the embeddings. The GUI may
        become unresponsive during this computation due to the processing required.
        """
        print("Calculating embedding, GUI may be unresponsive.")

        # Resize the image to 1024x1024 pixels, ensuring its 8-bit and applying antialiasing.
        img_1024 = transform.resize(
            self.img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        # Normalize the image pixel values to the range [0, 1].
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )

        # Convert the image to a PyTorch tensor and adjust its shape for the model.
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # Compute the embeddings with the MedSAM model's image encoder.
        with torch.no_grad():
            self.embedding = MEDSAM_MODEL.image_encoder(
                img_1024_tensor
            )  # (1, 256, 64, 64)

        print("Done calculating embedding.")


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()
