"""clotscape/ui/widgets/file_manager.py"""

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QWidget,
)


class FileManager(QWidget):
    """App File Manager widget"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        uploadButton = QPushButton(QIcon.fromTheme("document-open"), "Upload Image")
        uploadButton.setIconSize(QSize(24, 24))

        self.fileList = QListWidget()

        layout.addWidget(uploadButton)
        layout.addWidget(self.fileList)
        self.setLayout(layout)
