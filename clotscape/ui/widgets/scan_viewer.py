"""clotscape/ui/widgets/scan_viewer.py"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QWidget,
    QProgressBar,
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

        # Loading bar
        self.loading_bar = QProgressBar(self)
        self.loading_bar.setMaximum(0)
        self.loading_bar.setMinimum(0)
        self.loading_bar.setTextVisible(False)
        self.loading_bar.hide()

        layout.addWidget(label)
        layout.addWidget(self.image)
        layout.addWidget(self.loading_bar)

        self.setLayout(layout)

    def start_loading(self):
        """Start the loading animation."""
        self.loading_bar.show()

    def stop_loading(self):
        """Stop the loading animation."""
        self.loading_bar.hide()
