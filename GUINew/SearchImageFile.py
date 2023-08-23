from PyQt5.QtWidgets import QWidget, QHBoxLayout, QFormLayout, QLabel, QVBoxLayout, QCheckBox

from PyQt5.QtWidgets import QWidget

from GUI.GUIFunctions import *

class SearchImageView(QWidget):
    def __init__(self, width = 400, height = 400) -> None:
        super().__init__()
        self.image_width = width
        self.image_height = height
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0,0,0,0)
        self.content = QWidget()
        self.content.setLayout(self.content_layout)
        self.image = None
        self.content_desc = None

        self.layout_form = QFormLayout()
        self.setLayout(self.layout_form)
        self.layout_form.addWidget(self.content)

    def showImage(self, imagePath : str, img_data : ImageData = None):
        #TODO: PASS THROUGH ImageData and load the image before
        # Image ------------------------------------------
        if self.image:
            self.content_layout.removeWidget(self.image)
            self.image.deleteLater()
        self.image = QLabel(parent=self)
        self.image.setMaximumSize(self.image_width, self.image_height)
        # self.px = QPixmap(imagePath).scaled(width, height)
        img = img_data.orgImage
        self.org_img = numpy_to_pixmap(img).scaled(self.image_width, self.image_height)
        self.bb_img = numpy_to_pixmap(drawClasses(self.img_data, img.copy(), fontSize=img.shape[0]//600)).scaled(self.image_width, self.image_height)
        self.image.setPixmap(self.org_img)
        self.content_layout.addWidget(self.image)

        # Content desc ------------------------------------
        if self.content_desc:
            self.content_layout.removeWidget(self.content_desc)
            self.content_desc.deleteLater()
        self.content_desc = QWidget()
        self.content_desc_layout = QVBoxLayout()
        self.content_desc.setLayout(self.content_desc_layout)

        self.path_lbl = QLabel(text=f"Image path : {imagePath}")
        self.path_lbl.setWordWrap(True)
        self.content_desc_layout.addWidget(self.path_lbl)

        self.objects_lbl = QLabel(text=f"Objects : cat, cat, cat, cat")
        self.objects_lbl.setWordWrap(True)
        self.content_desc_layout.addWidget(self.objects_lbl)

        self.draw_bb = QCheckBox("Toggle bounding boxes", self)
        self.draw_bb.setCheckable(True)
        self.draw_bb.setChecked(False)
        self.draw_bb.clicked.connect(self.toggleBoundingBox)
        self.content_desc_layout.addWidget(self.draw_bb)
        self.content_layout.addWidget(self.content_desc)


    def toggleBoundingBox(self):
        self.image.setPixmap(self.bb_img if self.draw_bb.isChecked() else self.org_img)
        return
