"""app.py"""

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QWidget,
)

from .widgets import FileManager, ScanViewer, SegmentViewer


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set Window Title, Size, and Styling
        self.setWindowTitle("CT Scan Analysis")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(screen.width(), screen.height())

        # Main Font
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.setFont(font)

        # Use the new components
        fileManager = FileManager()
        mainImageViewer = ScanViewer()
        segmentViewer = SegmentViewer()

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(fileManager, 1)
        mainLayout.addWidget(mainImageViewer, 2)
        mainLayout.addWidget(segmentViewer, 1)

        centralWidget = QWidget(self)
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)
