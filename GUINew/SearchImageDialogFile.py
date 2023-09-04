import copy
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from GUINew.SearchImageFile import *
from GUINew.SearchParamsFile import SearchParams

from ImageAnalyzationModule.ImageAnalyzationFile import *
from ImageAnalyzationModule.ImageAnalyzationDataTypes import *
from GUI.GUIFunctions import *
from PyQt5.QtGui import QDoubleValidator


class SearchImageDialog(QDialog):
    def __init__(
        self,
        image_analyzation: ImageAnalyzation,
        image_width=400,
        image_height=400,
        options_only=False,
    ):
        super().__init__()
        self.seleted_object = -1
        self.setWindowTitle("Select image" if not options_only else "Settings")
        self.image_analyzation = image_analyzation
        self.has_image = False
        self.image_width = image_width
        self.image_height = image_height
        self.load_settings()

        self.main_layout = QHBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(self.main_layout)

        # Image preview -----------------------------------------------------
        if not options_only:
            self.image_preview_layout = QVBoxLayout()
            self.image_preview = QWidget()

            self.image_label = QLabel(parent=self.image_preview)
            self.image = QPixmap(self.search_params.imagePath).scaled(
                self.image_width, self.image_height
            )
            self.image_label.setPixmap(self.image)
            self.image_label.setMaximumSize(self.image_width, self.image_height)
            self.image_label.setMinimumSize(self.image_width, self.image_height)

            self.image_preview_layout.addWidget(self.image_label)

            self.image_button = QPushButton("Select image", parent=self.image_preview)
            self.image_button.clicked.connect(self.search_image)
            self.image_preview_layout.addWidget(self.image_button)

            self.image_preview.setLayout(self.image_preview_layout)
            self.main_layout.addWidget(self.image_preview)
        # Settings ----------------------------------------------------------
        self.settings_layout = QVBoxLayout()
        self.settings = QWidget()

        if not options_only:
            self.object_selection = QComboBox(self.settings)
            self.object_selection.addItem("All", userData=-1)
            self.object_selection.currentIndexChanged.connect(self.selection_change)
            self.settings_layout.addWidget(self.object_selection)

        self.compare_objects = QCheckBox("Compare objects", self.settings)
        self.compare_objects.setChecked(self.search_params.compareObjects)
        self.compare_objects.stateChanged.connect(self.prams_change)
        self.settings_layout.addWidget(self.compare_objects)

        self.compare_whole_images = QCheckBox("Compare whole images", self.settings)
        self.compare_whole_images.setChecked(self.search_params.compareWholeImages)
        self.compare_whole_images.stateChanged.connect(self.prams_change)
        self.settings_layout.addWidget(self.compare_whole_images)

        self.max_weight_reduciton = QCheckBox("Max weight reduction", self.settings)
        self.max_weight_reduciton.setChecked(self.search_params.maxWeightReduction)
        self.max_weight_reduciton.stateChanged.connect(self.prams_change)
        self.settings_layout.addWidget(self.max_weight_reduciton)

        self.contain_same_objects = QCheckBox(
            "Must contain same objects", self.settings
        )
        self.contain_same_objects.setChecked(self.search_params.containSameObjects)
        self.contain_same_objects.stateChanged.connect(self.prams_change)
        self.settings_layout.addWidget(self.contain_same_objects)

        self.conf_calc = QCheckBox(
            "Use object confidence in calculation", self.settings
        )
        self.conf_calc.setChecked(self.search_params.confidenceCalculation)
        self.conf_calc.stateChanged.connect(self.prams_change)
        self.settings_layout.addWidget(self.conf_calc)

        self.mag_calc = QCheckBox("Use magnitude calculation", self.settings)
        self.mag_calc.setChecked(self.search_params.magnitudeCalculation)
        self.mag_calc.stateChanged.connect(self.prams_change)
        self.settings_layout.addWidget(self.mag_calc)

        self.text_context = QCheckBox("Use image text context", self.settings)
        self.text_context.setChecked(self.search_params.textContext)
        self.text_context.stateChanged.connect(self.prams_change)
        self.settings_layout.addWidget(self.text_context)

        self.validator = QDoubleValidator()
        self.validator.setRange(0, 100, decimals=2)

        self.min_conf_widget = QWidget()
        self.min_conf_layout = QHBoxLayout()

        self.min_conf_label = QLabel("Minimum confidence for objects")
        self.min_conf_layout.addWidget(self.min_conf_label)

        self.min_conf = QLineEdit(
            str(self.search_params.minObjConf), parent=self.settings
        )
        self.min_conf.setValidator(self.validator)
        self.min_conf_layout.addWidget(self.min_conf)

        self.min_conf_widget.setLayout(self.min_conf_layout)
        self.settings_layout.addWidget(self.min_conf_widget)

        # Min weight line edit
        self.min_weight_widget = QWidget()
        self.min_weight_layout = QHBoxLayout()

        self.min_weight_label = QLabel("Minimum weight for objects")
        self.min_weight_layout.addWidget(self.min_weight_label)

        self.min_weight = QLineEdit(
            str(self.search_params.minObjWeight), parent=self.settings
        )
        self.min_weight.setValidator(self.validator)
        self.min_weight_layout.addWidget(self.min_weight)

        self.min_weight_widget.setLayout(self.min_weight_layout)
        self.settings_layout.addWidget(self.min_weight_widget)

        # text_context_weight
        self.text_context_weight_widget = QWidget()
        self.text_context_weight_widget_layout = QHBoxLayout()

        self.text_context_label = QLabel("Text context weight")
        self.text_context_weight_widget_layout.addWidget(self.text_context_label)

        self.text_context_line = QLineEdit(
            str(self.search_params.textContextWeight), parent=self.settings
        )
        self.text_context_line.setValidator(self.validator)
        self.text_context_weight_widget_layout.addWidget(self.text_context_line)

        self.text_context_weight_widget.setLayout(self.text_context_weight_widget_layout)
        self.settings_layout.addWidget(self.text_context_weight_widget)

        #Done

        self.okay_button = QPushButton("Okay")
        self.okay_button.clicked.connect(self.okay)
        self.settings_layout.addWidget(self.okay_button)

        self.settings.setLayout(self.settings_layout)
        self.main_layout.addWidget(self.settings)
        self.update_disabled()
        
    def load_settings(self):
        with open(".\\GUINew\\last_search.json", "r") as f:
            search_params_dict = json.load(fp=f)
            self.search_params = SearchParams()
            self.search_params.from_dict(search_params_dict)

    def okay(self):
        self.search_params.minObjConf = float(self.min_conf.text())
        self.search_params.minObjWeight = float(self.min_weight.text())
        self.search_params.textContextWeight = float(self.text_context_line.text())
        with open(".\\GUINew\\last_search.json", "w") as f:
            json.dump(self.search_params.get_dict(), fp=f)

        if self.has_image:
            self.accept()
        else:
            self.reject()
        return

    def search_image(self):
        options = QFileDialog.Options()
        photo_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photo",
            "",
            "Images (*.bmp *.pbm *.pgm *.gif *.sr *.ras *.jpeg *.jpg *.jpe *.jp2 *.tiff *.tif *.png *.mp4)",
            options=options,
        )

        if not photo_path:
            return

        self.search_params.imagePath = photo_path
        self.org_image = cv2.imread(photo_path)
        self.search_params.data = self.image_analyzation.getImageData(
            self.org_image,
            classesData=True,
            imageFeatures=True,
            objectsFeatures=True,
            returnOriginalImage=False,
            classesConfidence=0.35,
        )
        self.org_image = cv2.cvtColor(self.org_image, cv2.COLOR_BGR2RGB)
        self.search_params.data.orgImage = (
            self.org_image
        )  # The reson returnOriginalImage is False is because the color correction above

        while self.object_selection.count() > 1:
            self.object_selection.removeItem(1)

        for i, c in enumerate(self.search_params.data.classes):
            self.object_selection.addItem(c.className, i)

        self.update_image()

        self.has_image = True

    def update_image(self):
        self.bb_image = drawClasses(
            self.search_params.data,
            self.org_image.copy(),
            index=self.search_params.selectedIndex,
        )
        self.bb_image_px = numpy_to_pixmap(self.bb_image)
        self.image = self.bb_image_px.scaled(self.image_width, self.image_height)
        self.image_label.setPixmap(self.image)

    def selection_change(self):
        self.search_params.selectedIndex = (
            None
            if self.object_selection.currentData() == -1
            else self.object_selection.currentData()
        )
        print(self.search_params.selectedIndex)
        disable_controls = self.search_params.selectedIndex is not None
        self.compare_objects.setDisabled(disable_controls)
        self.compare_whole_images.setDisabled(disable_controls)
        self.max_weight_reduciton.setDisabled(disable_controls)
        self.contain_same_objects.setDisabled(disable_controls)
        # self.conf_calc.setDisabled(disable_controls)
        # self.mag_calc.setDisabled(disable_controls)
        self.update_image()
        return

    def prams_change(self, checkbox):
        self.search_params.compareObjects = self.compare_objects.isChecked()
        self.search_params.compareWholeImages = self.compare_whole_images.isChecked()
        self.search_params.maxWeightReduction = self.max_weight_reduciton.isChecked()
        self.search_params.containSameObjects = self.contain_same_objects.isChecked()
        self.search_params.confidenceCalculation = self.conf_calc.isChecked()
        self.search_params.magnitudeCalculation = self.mag_calc.isChecked()
        self.search_params.textContext = self.text_context.isChecked()
        self.update_disabled()
        return
    def update_disabled(self):
        self.text_context_line.setDisabled(not self.text_context.isChecked())
