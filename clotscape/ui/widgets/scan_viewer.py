"""clotscape/ui/widgets/scan_viewer.py"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QWidget,
)

from .image import Image


class ScanViewer(QWidget):
    """App CT Scan Viewer widget"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("Scan Viewer")
        label.setFont(QFont("Arial", 12, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)

        # Add the Image widget
        self.image = Image()

        layout.addWidget(label)
        layout.addWidget(self.image)

        self.setLayout(layout)

    def display_image(self, image_path: str):
        """Display an image in the viewer."""
        self.image.set_image(image_path)
