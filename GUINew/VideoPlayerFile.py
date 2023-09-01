from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QDir, Qt, QUrl, QSize, QPoint, pyqtSignal
import PyQt5.QtCore as QtCore
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QCursor

class VideoPlayer(QWidget):
    def __init__(self, fileName="", data : list = None,parent=None, item_size = 200):
        super(VideoPlayer, self).__init__(parent)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.collum_number = self.width() // item_size
        self.item_size = item_size
        btnSize = QSize(16, 16)
        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        self.positionSlider.sliderPressed.connect(self.onSliderClicked)
        
        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        #Adding all the video frames

        self.scroll_area = QScrollArea()
        self.scroll_area.setContentsMargins(0,0,0,0)
        self.content_frames = QWidget()
        self.content_layout = QGridLayout()
        self.content_layout.setRowMinimumHeight(0, item_size)
        self.content_layout.setSpacing(0)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.old_resize_frames = self.resizeEvent
        self.scroll_area.resizeEvent = self.resize_frames

        data.sort(key=lambda x : int(x[1]))

        for ed in enumerate(data):
            d = ed[1]
            i = ed[0]
            vpi = VideoPlayerItem(d[1], d[0], size=(item_size, item_size))
            vpi.clicked.connect(self.item_click_position_change) 
            self.content_layout.addWidget(vpi, i // self.collum_number, i % self.collum_number)

        self.content_frames.setLayout(self.content_layout)
        self.scroll_area.setWidget(self.content_frames)
        self.scroll_area.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget, stretch= 6)
        layout.addLayout(controlLayout, stretch= 1)
        layout.addWidget(self.statusBar, stretch= 2)
        layout.addWidget(self.scroll_area, stretch= 3)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")
        
        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(fileName)
            self.play()
        self.item_click_position_change(int(data[0][1]))

    @QtCore.pyqtSlot(int)
    def item_click_position_change(self, frame_num : int):
        self.mediaPlayer.setPosition((int(frame_num) // 30) * 1000)

    def resize_frames(self, event):
        self.old_resize_frames(event)
        self.content_frames.setMinimumWidth(self.scroll_area.width())
        old_collum = self.collum_number
        self.collum_number = max(self.scroll_area.width() // self.item_size, 1)
        if old_collum == self.collum_number:
            return
        
        widgets = [
            (i, self.content_layout.itemAt(i).widget())
            for i in range(self.content_layout.count())
        ]
        while self.content_layout.count() > 0:
            item = self.content_layout.itemAt(0)
            if item.widget():
                self.content_layout.removeWidget(item.widget())
            else:
                self.content_layout.removeItem(item)
        
        for i, widget in widgets:
            self.content_layout.addWidget(
            widget, i // self.collum_number, i % self.collum_number
            )

    def abrir(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Selecciona los mediose",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(fileName)
            self.play()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            #self.mediaPlayer.setPosition(500)
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(int(position))

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())


    def onSliderClicked(self):
        click_position = QCursor.pos().x() - self.positionSlider.mapToGlobal(QPoint(0, 0)).x()
        max_position = self.positionSlider.width()
        value = (click_position / max_position) * self.positionSlider.maximum()
        self.setPosition(value)
 
    def closeEvent(self, event):
        self.mediaPlayer.stop()
        event.accept()

    def closePlayer(self):
        self.mediaPlayer.stop()

class VideoPlayerItem(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, frame_num = 0, image : QPixmap = None, size = (200, 200)):
        super().__init__()
        self.frame_num = frame_num
        self.image_label = QLabel()
        self.image_label.setPixmap(image.scaled(size[0], size[1]))
        self.main_layot = QHBoxLayout()
        self.main_layot.addWidget(self.image_label)
        self.setLayout(self.main_layot)
        self.setMaximumWidth(size[0])
        self.setMinimumWidth(size[0])
        self.setMaximumHeight(size[1])
        self.setMinimumHeight(size[1])
        self.mouseDoubleClickEvent = self.double_click_event
        self.setContentsMargins(0,0,0,0)
    
    def double_click_event(self, event):
        self.clicked.emit(int(self.frame_num))
