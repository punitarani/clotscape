"""clotscape/ui/widgets/scan_viewer.py"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QWidget,
)


class ScanViewer(QWidget):
    """App CT Scan Viewer widget"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("Main Image Viewer")
        label.setFont(QFont("Arial", 12, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)

        self.imageView = QLabel("(Selected image will be displayed here)")
        self.imageView.setAlignment(Qt.AlignCenter)

        layout.addWidget(label)
        layout.addWidget(self.imageView)
        self.setLayout(layout)
