"""clotscape/ui/widgets/scan_viewer.py"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QWidget,
)


class SegmentViewer(QWidget):
    """App CT Scan Segments Viewer widget"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("Segments Viewer")
        label.setFont(QFont("Arial", 12, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)

        self.segmentView = QLabel(
            "(Segments of the selected image will be displayed here)"
        )
        self.segmentView.setAlignment(Qt.AlignCenter)

        layout.addWidget(label)
        layout.addWidget(self.segmentView)
        self.setLayout(layout)
