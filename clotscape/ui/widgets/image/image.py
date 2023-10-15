"""clotscape/ui/widgets/image/image.py"""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap, QImageReader
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QScrollArea,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QApplication,
)


class Image(QWidget):
    """A widget to display an image."""

    def __init__(self):
        super().__init__()

        # The path to the image to be displayed
        self._image_path = None

        # Set up the layout and widgets
        self.layout = QVBoxLayout()

        # Scroll area to allow for zoomed-in images to be scrolled
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        # The actual label that will contain the image
        self.image_label = QLabel()
        self.scroll_area.setWidget(self.image_label)

        # Zooming variables
        self.zoom_factor = 1.0
        self.max_zoom_factor = 3.0
        self.min_zoom_factor = 0.1
        self.zoom_step = 0.1

        # Add the Zoom In, Zoom Out, and Reset buttons
        self.button_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("+")
        self.zoom_out_button = QPushButton("-")
        self.reset_button = QPushButton("Reset Zoom")

        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.reset_button.clicked.connect(self.reset_zoom)

        self.button_layout.addWidget(self.zoom_in_button)
        self.button_layout.addWidget(self.zoom_out_button)
        self.button_layout.addWidget(self.reset_button)
        self.layout.addLayout(self.button_layout)

        # Add the scroll area to the layout
        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)

    def set_image(self, image_path: str):
        """Set the image to be displayed."""
        self._image_path = image_path
        self.zoom_factor = 1.0
        self._update_image()

    def _update_image(self):
        """Internal method to handle image updating."""
        if not self._image_path:
            return

        image_reader = QImageReader(self._image_path)

        # Get the size of the widget
        target_width = int(0.925 * self.width() * self.zoom_factor)
        target_height = int(0.925 * self.height() * self.zoom_factor)

        # Calculate image's aspect ratio
        img_width = image_reader.size().width()
        img_height = image_reader.size().height()
        img_aspect_ratio = img_width / img_height

        # Determine the target dimensions based on the image's aspect ratio
        if img_aspect_ratio > 1:  # Image is wider
            target_height = int(target_width / img_aspect_ratio)
        else:  # Image is taller or square
            target_width = int(target_height * img_aspect_ratio)

        # Set the scaled size to the QImageReader
        image_reader.setScaledSize(QSize(target_width, target_height))

        pixmap = QPixmap.fromImageReader(image_reader)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

        # Center the image
        self.image_label.setAlignment(Qt.AlignCenter)

    def clear_image(self):
        """Clear the currently displayed image."""
        self.image_label.clear()

    def resizeEvent(self, event):
        """Handle the widget's resize event to update the image."""
        self._update_image()
        super().resizeEvent(event)

    def zoom_in(self):
        """Zoom in on the image."""
        if self.zoom_factor < self.max_zoom_factor:
            self.zoom_factor += self.zoom_step
            self._update_image()

    def zoom_out(self):
        """Zoom out of the image."""
        if self.zoom_factor > self.min_zoom_factor:
            self.zoom_factor -= self.zoom_step
            self._update_image()

    def reset_zoom(self):
        """Reset the zoom to the original size."""
        self.zoom_factor = 1.0
        self._update_image()

    def wheelEvent(self, event):
        """Handle the mouse wheel event for zooming."""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            if event.angleDelta().y() > 0:  # Mouse wheel scrolled up
                self.zoom_in()
            else:  # Mouse wheel scrolled down
                self.zoom_out()

    def keyPressEvent(self, event):
        """Handle key press events for zooming."""
        if event.key() == Qt.Key_Plus and event.modifiers() == Qt.ControlModifier:
            self.zoom_in()
        elif event.key() == Qt.Key_Minus and event.modifiers() == Qt.ControlModifier:
            self.zoom_out()
        elif event.key() == Qt.Key_0 and event.modifiers() == Qt.ControlModifier:
            self.reset_zoom()
