"""clotscape/ui/widgets/file_manager.py"""

import re
from pathlib import Path

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QWidget,
    QHBoxLayout,
    QMessageBox,
    QInputDialog,
    QFileDialog,
)

from clotscape import Project


class FileManager(QWidget):
    """App File Manager widget"""

    project_changed = Signal(Project)

    def __init__(self, project: Project = None):
        """
        FileManager constructor

        Args:
            project (Project, optional): Project to load. Defaults to None.
        """
        super().__init__()

        self.project: Project = project

        self.layout = QVBoxLayout()

        self.topButtonLayout = QHBoxLayout()

        # Open Project Button
        self.openProjectButton = QPushButton(
            QIcon.fromTheme("folder-open"), "Open Project"
        )
        self.openProjectButton.setIconSize(QSize(24, 24))
        self.openProjectButton.clicked.connect(self.open_project)

        # Create Project Button
        self.createProjectButton = QPushButton(
            QIcon.fromTheme("document-new"), "Create Project"
        )
        self.createProjectButton.setIconSize(QSize(24, 24))
        self.createProjectButton.clicked.connect(self.create_project)

        # Close Project Button
        self.closeProjectButton = QPushButton(
            QIcon.fromTheme("window-close"), "Close Project"
        )
        self.closeProjectButton.setIconSize(QSize(24, 24))
        self.closeProjectButton.clicked.connect(self.close_project)

        self.topButtonLayout.addWidget(self.openProjectButton)
        self.topButtonLayout.addWidget(self.createProjectButton)
        self.topButtonLayout.addWidget(self.closeProjectButton)

        self.layout.addLayout(self.topButtonLayout)

        # Upload Image Button
        self.uploadButton = QPushButton(
            QIcon.fromTheme("document-open"), "Upload Image"
        )
        self.uploadButton.setIconSize(QSize(24, 24))
        # TODO: Connect this button to the appropriate function
        # self.uploadButton.clicked.connect(self.upload_image)

        self.fileList = QListWidget()

        self.layout.addWidget(self.uploadButton)
        self.layout.addWidget(self.fileList)

        self.setLayout(self.layout)

        self.update_display()

    def update_display(self):
        """Update the display based on whether a project is loaded."""
        if self.project:
            self.createProjectButton.hide()
            self.closeProjectButton.show()
            self.uploadButton.show()
            self.fileList.show()
            # TODO: Populate the fileList with the project's files
        else:
            self.createProjectButton.show()
            self.closeProjectButton.hide()
            self.uploadButton.hide()
            self.fileList.hide()

    def open_project(self):
        """Open an existing project."""
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Clotscape Project",
            "",
            "Clotscape Files (*.clotscape);;All Files (*)",
        )
        if fileName:
            try:
                self.project = Project.load(Path(fileName))
                self.project_changed.emit(self.project)
                self.update_display()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load the project: {e}")

    def create_project(self) -> None:
        """Create a new project."""
        name, ok = QInputDialog.getText(
            self, "Create Project", "Project Name (alphabets and '-'):"
        )

        if ok and name and self.validate_project_name(name):
            folder = QFileDialog.getExistingDirectory(self, "Select Parent Directory")

            if folder:
                self.project = Project.create(name, Path(folder))
                self.project_changed.emit(self.project)
                self.update_display()
        else:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Project name is invalid. Use only alphabets and '-'.",
            )

    def close_project(self) -> None:
        """Close the current project."""
        self.project = None
        self.project_changed.emit(self.project)
        self.fileList.clear()
        self.update_display()

    @staticmethod
    def validate_project_name(name: str) -> bool:
        """
        Validate project name.
        Allowed characters are alphabets and '-'.

        Args:
            name (str): Project name

        Returns:
            bool: True if valid, False otherwise
        """
        return bool(re.match(r"^[A-Za-z-]+$", name))
