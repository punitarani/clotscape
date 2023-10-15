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
        self.uploadButton.clicked.connect(self.upload_image)

        self.file_list = QListWidget()

        self.layout.addWidget(self.uploadButton)
        self.layout.addWidget(self.file_list)

        self.setLayout(self.layout)

        self.update_display()

    def update_display(self):
        """Update the display based on whether a project is loaded."""
        if self.project:
            self.createProjectButton.hide()
            self.closeProjectButton.show()
            self.uploadButton.show()
            self.file_list.show()

            # Populate the file_list with the project's files
            self.file_list.clear()
            for name, path in self.project.images.items():
                self.file_list.addItem(f"{name} ({path})")
        else:
            self.createProjectButton.show()
            self.closeProjectButton.hide()
            self.uploadButton.hide()
            self.file_list.hide()

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

        if ok and name and self.validate_name(name):
            folder = QFileDialog.getExistingDirectory(self, "Select Parent Directory")

            if folder:
                self.project = Project.create(name, Path(folder).joinpath(name))
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
        self.file_list.clear()
        self.update_display()

    def upload_image(self):
        """Upload a new image to the project."""
        filePath, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Image",
            "",
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)",
        )

        if filePath:
            # Validate image extension
            ext = Path(filePath).suffix.lower()
            if ext not in [".png", ".jpg", ".jpeg"]:
                QMessageBox.warning(
                    self,
                    "Invalid Image",
                    "Only .jpg, .jpeg, and .png images are allowed.",
                )
                return

            # Ask user for image name
            image_name, ok = QInputDialog.getText(
                self,
                "Image Name",
                "Enter name for the image (leave empty to use file name):",
            )

            # If user didn't provide a name or if the name is not valid, use the file name
            if not ok or not image_name or not self.validate_name(image_name):
                image_name = Path(filePath).stem

            # Assuming the Project class has a method to add new images
            # You'll need to implement this
            self.project.add_image(image_name, Path(filePath))

            # Update the file_list
            self.file_list.addItem(f"{image_name} ({filePath})")

    @staticmethod
    def validate_name(name: str) -> bool:
        """
        Validate name, Allowed characters are alphabets and '-'.

        Args:
            name (str): Project name

        Returns:
            bool: True if valid, False otherwise
        """
        return bool(re.match(r"^[A-Za-z-]+$", name))
