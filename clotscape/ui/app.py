"""app.py"""

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QWidget,
)

from .widgets import FileManager, ScanViewer, SegmentViewer

from clotscape import Project


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.project: Project | None = None

        # Set Window Title, Size, and Styling
        self.setWindowTitle("Clotscape")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(screen.width(), screen.height())

        # Main Font
        font = QFont()
        font.setFamily("Roboto")
        font.setPointSize(12)
        self.setFont(font)

        # Initialize the components
        self.file_manager = FileManager()
        self.scan_viewer = ScanViewer()
        self.segment_viewer = SegmentViewer()

        # Set up the connections
        self.file_manager.project_changed.connect(self.set_project)

        self.main_layout = QHBoxLayout()

        self.central_widget = QWidget(self)
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        self.update_display()

    def update_display(self):
        """Update the display based on whether a project is loaded."""
        for i in reversed(range(self.main_layout.count())):
            widget = self.main_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)  # This removes the widget from layout

        if self.project:
            self.main_layout.addWidget(self.file_manager, 1)
            self.main_layout.addWidget(self.scan_viewer, 2)
            self.main_layout.addWidget(self.segment_viewer, 1)
        else:
            self.main_layout.addWidget(self.file_manager, 1)

    def set_project(self, project: Project):
        """Set the project and update the display."""
        self.project = project
        self.update_display()
