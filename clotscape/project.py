"""clotscape/project.py"""

import json
from pathlib import Path
from uuid import UUID, uuid4


class Project:
    """Project Class"""

    def __init__(
        self, project_id: UUID, name: str, folder: Path, images: dict[str, Path]
    ):
        """Initialize Project"""

        self.id: UUID = project_id
        self.name: str = name
        self.folder: Path = folder

        self.images: dict[str, Path] = images

    @classmethod
    def create(cls, name: str, folder: Path) -> "Project":
        """
        Create a new Project

        Args:
            name (str): Project name
            folder (Path): Folder to save project

        Returns:
            Project: New Project
        """

        project_id = uuid4()
        images = {}

        project = cls(project_id=project_id, name=name, folder=folder, images=images)
        project.save()

        return project

    @classmethod
    def load(cls, config: Path) -> "Project":
        """
        Load Project from config file

        Args:
            config (Path): Path to config file(*.clotscape)
        """

        with open(config, "r") as f:
            data = json.load(f)

        project_id = UUID(data["id"])
        name = data["name"]
        images = {
            name: config.parent.joinpath("images", path)
            for name, path in data["images"].items()
        }

        return cls(
            project_id=project_id, name=name, folder=config.parent, images=images
        )

    def save(self) -> None:
        """Save Project to config file in the specified folder."""

        data = {
            "id": str(self.id),
            "name": self.name,
            "images": {
                name: path.relative_to(self.folder.joinpath("images"))
                for name, path in self.images.items()
            },
        }

        with open(self.folder, "w") as f:
            json.dump(data, f, indent=4)
