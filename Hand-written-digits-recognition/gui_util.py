from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *


# 定义在标签控件上显示识别结果的函数
# 参数：img - 输入图像，label - 目标标签控件，w - 显示宽度，h - 显示高度
def show_image_on_label(img, label, w, h):
    # 检查标签控件是否有效
    if label is None:
        return
    # 将BGR格式转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 调整图像尺寸
    img = cv2.resize(img, (w, h))
    # 将numpy数组转换为QImage对象
    img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
    # 将QImage转换为QPixmap
    pixmap = QPixmap.fromImage(img)
    # 将QPixmap设置到QLabel控件上显示
    label.setPixmap(pixmap.scaled(w, h, aspectRatioMode=Qt.IgnoreAspectRatio))


# 定义获取摄像头视频流的函数
# 返回：视频捕获对象
def camera_cap():
    # 设置视频宽度和高度
    width, height = 875, 625
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    # 设置帧宽度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # 设置帧高度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 返回视频捕获对象
    return cap
