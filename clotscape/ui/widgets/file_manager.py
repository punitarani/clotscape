"""clotscape/ui/widgets/file_manager.py"""

import json
import uuid
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QWidget,
    QHBoxLayout,
    QFileDialog,
)


class FileManager(QWidget):
    """App File Manager widget"""

    def __init__(self, folder: Path = None):
        """
        FileManager constructor

        Args:
            folder (Path, optional): The folder to use for the file manager.
                Defaults to None, which will use the user's home directory.
        """
        super().__init__()

        self.folder: Path = folder
        if self.folder is None:
            self.folder = Path.home().joinpath("clotscape")

        self.selected_file_path = None  # Class variable to store selected file path

        layout = QVBoxLayout()

        # Create top button layout
        topButtonLayout = QHBoxLayout()

        # Open Project Button
        openProjectButton = QPushButton(QIcon.fromTheme("folder-open"), "Open Project")
        openProjectButton.setIconSize(QSize(24, 24))
        openProjectButton.clicked.connect(self.open_project)

        # Create Project Button
        createProjectButton = QPushButton(
            QIcon.fromTheme("document-new"), "Create Project"
        )
        createProjectButton.setIconSize(QSize(24, 24))
        createProjectButton.clicked.connect(self.create_project)

        # Add open and save buttons to the top button layout
        topButtonLayout.addWidget(openProjectButton)
        topButtonLayout.addWidget(createProjectButton)

        # Add the top button layout to the main layout
        layout.addLayout(topButtonLayout)

        # Upload Image Button
        uploadButton = QPushButton(QIcon.fromTheme("document-open"), "Upload Image")
        uploadButton.setIconSize(QSize(24, 24))

        # File List
        self.fileList = QListWidget()

        # Add upload button and file list to the main layout
        layout.addWidget(uploadButton)
        layout.addWidget(self.fileList)

        self.setLayout(layout)

    def create_project(self):
        """Create a project to a .clot.scape file based on the folder the user selects."""

        # Open a directory picker dialog
        selected_folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save Project"
        )
        if not selected_folder:
            return

        # Generate unique ID
        project_id = str(uuid.uuid4())

        # Get folder name for the project name and filename
        folder_path = Path(selected_folder)
        project_name = folder_path.name
        file_path = folder_path.joinpath(f"{project_name}.clot.scape")

        # Construct the data in the specified format
        data = {"id": project_id, "name": project_name, "images": {}}

        # Assuming the images are stored in the same format in self.fileList
        for index in range(self.fileList.count()):
            item_text = self.fileList.item(index).text()
            image_name, image_path = item_text.split(": ")
            # Extracting the image filename from the absolute path
            relative_image_path = Path(image_path).name
            data["images"][image_name] = relative_image_path

        # Save the data to the .clot.scape file
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def open_project(self):
        """Open a file explorer to select a .clot.scape project file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project File", "", "ClotScape Files (*.clot.scape)"
        )
        if file_path:
            self.selected_file_path = Path(file_path)
            self._load_images_from_config()
            self.fileList.clear()

    def _load_images_from_config(self):
        """Load and display images from the selected .clot.scape config file."""
        with self.selected_file_path.open() as file:
            config = json.load(file)
            image_dict = config.get("images", {})
            parent_dir = self.selected_file_path.parent

            for image_name, relative_image_path in image_dict.items():
                absolute_image_path = parent_dir.joinpath(relative_image_path)
                # Display the image name in the file manager (QListWidget)
                self.fileList.addItem(f"{image_name}: {absolute_image_path}")
