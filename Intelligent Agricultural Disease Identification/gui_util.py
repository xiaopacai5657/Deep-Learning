from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *


# 定义在标签上显示识别结果的函数
def show_image_on_label(img, label, w, h):
    if label is None:
        return
    # 将BGR格式转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 调整图像大小
    img = cv2.resize(img, (w, h))
    # 将numpy数组转换为QImage
    img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
    # 将QImage转换为QPixmap
    pixmap = QPixmap.fromImage(img)
    # 将QPixmap设置到QLabel上
    label.setPixmap(pixmap.scaled(w, h, aspectRatioMode=Qt.IgnoreAspectRatio))
