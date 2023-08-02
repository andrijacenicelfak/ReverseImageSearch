import sys
from GLUE import GUI
from SqliteDB import ImageDB
from ImageAnalyzation import ImageAnalyzation
from PyQt5.QtWidgets import (
    QApplication
)

img_analyzer=ImageAnalyzation("yolov8s.pt","cuda")
img_db=ImageDB()

app = QApplication(sys.argv)
app.setStyle('Fusion')
window = GUI(img_db,img_analyzer)
window.show()
sys.exit(app.exec_())
