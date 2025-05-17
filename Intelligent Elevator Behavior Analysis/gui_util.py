from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *


# 定义在标签上显示识别结果的函数
def show_image_on_label(img, label, w, h):
    if label is None:
        return
        # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image
    img = cv2.resize(img, (w, h))
    # Convert numpy array to QImage
    img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(img)
    # Set QPixmap onto QLabel
    label.setPixmap(pixmap.scaled(w, h, aspectRatioMode=Qt.IgnoreAspectRatio))
