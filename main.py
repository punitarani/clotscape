"""GUI entry point."""

import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QWidget,
)


class FileManager(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        uploadButton = QPushButton(QIcon.fromTheme("document-open"), "Upload Image")
        uploadButton.setIconSize(QSize(24, 24))

        self.fileList = QListWidget()

        layout.addWidget(uploadButton)
        layout.addWidget(self.fileList)
        self.setLayout(layout)


class MainImageViewer(QWidget):
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


class SegmentViewer(QWidget):
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


class CTScanApp(QMainWindow):
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
        mainImageViewer = MainImageViewer()
        segmentViewer = SegmentViewer()

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(fileManager, 1)
        mainLayout.addWidget(mainImageViewer, 2)
        mainLayout.addWidget(segmentViewer, 1)

        centralWidget = QWidget(self)
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = CTScanApp()
    window.show()

    sys.exit(app.exec())
