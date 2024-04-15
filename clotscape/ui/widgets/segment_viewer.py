"""clotscape/ui/widgets/scan_viewer.py"""

from pathlib import Path

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class SegmentViewer(QWidget):
    """App CT Scan Segments Viewer widget"""

    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.original_image_label = QLabel()
        self.mask_image_label = QLabel()

        self.layout.addWidget(self.original_image_label)
        self.layout.addWidget(self.mask_image_label)

        self.setLayout(self.layout)

    def display_segmented_image(self, original_image_path: Path, mask_image: QPixmap):
        self.original_image_label.setPixmap(QPixmap(original_image_path))
        self.mask_image_label.setPixmap(mask_image)
