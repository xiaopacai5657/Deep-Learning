from PyQt5.QtWidgets import *
import time
import os
from Pre_treatment import get_number as g_n
from Pre_treatment import get_number_2 as g_n_2
from Pre_treatment import get_roi
import predict as pt
from Pre_treatment import softmax
from gui_util import *
from qfluentwidgets import InfoBar, InfoBarPosition
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QStackedWidget, QLabel
from qfluentwidgets import SegmentedWidget


# 手写数字识别
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.net = pt.get_net()  # 加载预训练的神经网络模型
        self.btn_exit = None  # 退出系统按钮
        self.statusbar = None  # 状态栏
        self.handwriting = None  # 写字板
        self.btn_hw_recog1 = None  # 识别数字按钮
        self.btn_hw_clear1 = None  # 清空写字板按钮
        self.btn_hw_save1 = None  # 保存字迹按钮
        self.label_result1 = None  # 识别结果标签
        self.label_num1 = None  # 结果显示标签
        self.label_file2 = None  # 图片显示标签
        self.btn_file2 = None  # 上传文件按钮
        self.current_image_path = None  # 当前所选文件
        self.btn_file_recog2 = None  # 识别数字按钮
        self.btn_flie_clear2 = None  # 清空图片按钮
        self.label_result2 = None  # 识别结果标签
        self.label_num2 = None  # 结果显示标签
        self.label_camera3 = None  # 摄像头画面显示标签
        self.camera_running = False  # 摄像头标志位
        self.btn_open_camera3 = None  # 开启摄像头按钮
        self.btn_save_camera3 = None  # 保存视频帧按钮
        self.btn_close_camera3 = None  # 关闭摄像头按钮
        self.label_result3 = None  # 识别结果标签
        self.label_num3 = None  # 结果显示标签
        # 设置全屏显示
        # self.showFullScreen()
        # 隐藏窗口标题栏
        self.setWindowFlags(Qt.FramelessWindowHint)
        # 设置窗口图标
        self.setWindowIcon(QIcon("./ui_img/icon.png"))
        # 设置窗口大小（固定大小）
        self.setFixedSize(1280, 720)
        # 设置背景图片
        self.setStyleSheet("""
            QMainWindow {
                background-image: url('./ui_img/background.jpg'); /* 替换为你的图片路径 */
                background-repeat: no-repeat;             /* 不重复 */
                background-position: center;              /* 图片居中 */
                background-size: cover;                   /* 背景图覆盖整个窗口 */
            }
        """)
        # 查看一下窗口尺寸
        print(self.width, self.height)
        # 加载setup_ui函数
        self.setup_ui()

    def setup_ui(self):
        # 创建退出系统按钮
        self.btn_exit = QPushButton(" 退 出 系 统 ", self)  # 添加按钮文本
        # 设置退出系统按钮的样式
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 - 透明 */
                color: black;              /* 文字颜色 - 黑色 */
                border-radius: 5px;        /* 圆角半径 */
                font: bold 22px;           /* 字体加粗，22像素 */
                padding: 5px;              /* 内边距 */
                text-align: center;        /* 文字居中  */
                border: 2px solid black; /* 边框属性 */
            }
            QPushButton:hover {
                background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
            }
            QPushButton:pressed {
                background-color: #B8D4FF; /* 按下时的更深蓝色 */
                color: black;           /* 按下时文字颜色加深 */
                padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                padding-top: 3px;
                border: 2px solid black; /* 按下时边框颜色 */
            }
        """)
        # 设置按钮的位置和大小 setGeometry(x, y, width, height)
        self.btn_exit.setGeometry(1050, 35, 200, 50)
        # 连接按钮的点击信号到槽函数
        self.btn_exit.clicked.connect(self.onExitButtonClicked)

        # 创建 SegmentedWidget
        self.pivot = SegmentedWidget(self)
        # 设置 SegmentedWidget 样式
        self.pivot.setStyleSheet("""
            SegmentedWidget {
                background-color: rgba(255, 255, 255, 0.5); /* 整体背景半透明白色 */
                border: 7px solid rgba(158, 201, 254, 1); /* 半透明白色边框 */
                border-radius: 10px; /* 圆角 */
            }

            /* 普通状态的选项 */
            QPushButton {
                background-color: rgba(255, 255, 255, 0.5); /* 选项背景半透明 */
                color: black;  /* 文字颜色黑色 */
                font:  20px;  /* 文字属性 */
                padding: 10px 25px; /* 适当增加内边距，使按钮更大 */
                border-radius: 5px; /* 圆角半径 */
                text-align: center; /* 文字居中  */
                transition: background-color 0.2s ease-in-out; /* 平滑过渡 */
            }

            /* 鼠标悬停时 */
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.5); /* 背景变亮 */
            }

            /* 选中状态 */
            QPushButton:selected {
                background-color: #00aaff;  /* 选中项为亮蓝色 */
                color: white; /* 选中文字变白 */
                font-weight: bold;
            }
        """)
        self.pivot.setFixedSize(400, 40)  # 设定固定大小
        self.pivot.move(850, 105)  # 设定绝对位置

        # 创建 QStackedWidget
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.setFixedSize(1280, 570)  # 设定固定大小
        self.stackedWidget.move(0, 150)  # 设定绝对位置

        # 创建子界面
        self.handwriting_interface = QLabel(self)
        self.camera_interface = QLabel(self)
        self.file_interface = QLabel(self)

        # 添加子界面
        self.addSubInterface(self.handwriting_interface, 'handwritingInterface', '手写板')
        self.addSubInterface(self.camera_interface, 'cameraInterface', '摄像头')
        self.addSubInterface(self.file_interface, 'fileInterface', '本地文件')

        # 设置默认界面
        self.stackedWidget.setCurrentWidget(self.handwriting_interface)
        self.pivot.setCurrentItem(self.handwriting_interface.objectName())

        # 连接切换信号
        def on_pivot_changed(k):
            # 切换界面
            target_widget = self.findChild(QWidget, k)
            if target_widget:
                self.stackedWidget.setCurrentWidget(target_widget)
                # 执行额外的槽函数
                self.on_interface_changed(target_widget)

        self.pivot.currentItemChanged.connect(on_pivot_changed)
        # 连接 QStackedWidget 的 currentChanged 信号
        self.stackedWidget.currentChanged.connect(self.on_stack_changed)

        # # 连接切换信号
        # self.pivot.currentItemChanged.connect(
        #     lambda k: self.stackedWidget.setCurrentWidget(self.findChild(QWidget, k))
        # )

        # 创建手写板控件
        self.handwriting = HandwritingWidget(self.handwriting_interface)
        self.handwriting.setGeometry(50, 5, 800, 520)

        # 创建识别数字按钮
        self.btn_hw_recog1 = QPushButton("识 别 数 字", self.handwriting_interface)
        self.btn_hw_recog1.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                        border: 2px solid black; /* 边框属性 */
                    }
                    QPushButton:hover {
                        background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                        border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
                    }
                    QPushButton:pressed {
                        background-color: #B8D4FF; /* 按下时的更深蓝色 */
                        color: black;           /* 按下时文字颜色加深 */
                        padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                        padding-top: 3px;
                        border: 2px solid black; /* 按下时边框颜色 */
                    }
                """)
        self.btn_hw_recog1.setGeometry(965, 40, 200, 50)
        self.btn_hw_recog1.clicked.connect(self.recognize_handwriting)

        # 创建清空写字板按钮
        self.btn_hw_clear1 = QPushButton("清 空 写 字 板", self.handwriting_interface)
        self.btn_hw_clear1.setStyleSheet("""
            QPushButton {
                background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                color: black;              /* 文字颜色 - 黑色 */
                border-radius: 5px;        /* 圆角半径 */
                font: bold 22px;           /* 字体加粗，22像素 */
                padding: 5px;              /* 内边距 */
                text-align: center;        /* 文字居中  */
                border: 2px solid black; /* 边框属性 */
            }
            QPushButton:hover {
                background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
            }
            QPushButton:pressed {
                background-color: #B8D4FF; /* 按下时的更深蓝色 */
                color: black;           /* 按下时文字颜色 */
                padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                padding-top: 3px;
                border: 2px solid black; /* 按下时边框颜色 */
            }
        """)
        self.btn_hw_clear1.setGeometry(965, 120, 200, 50)
        self.btn_hw_clear1.clicked.connect(
            lambda: [
                self.handwriting.clear(),
                self.label_result1.clear(),
                self.label_num1.setText("识别结果:  准确度:  "),
                self.createSuccessInfoBar("清除成功", "写字板内容已清空")
            ]
        )

        # 创建保存字迹按钮
        self.btn_hw_save1 = QPushButton("保 存 字 迹", self.handwriting_interface)
        self.btn_hw_save1.setStyleSheet("""
            QPushButton {
                background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                color: black;              /* 文字颜色 - 黑色 */
                border-radius: 5px;        /* 圆角半径 */
                font: bold 22px;           /* 字体加粗，22像素 */
                padding: 5px;              /* 内边距 */
                text-align: center;        /* 文字居中  */
                border: 2px solid black; /* 边框属性 */
            }
            QPushButton:hover {
                background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
            }
            QPushButton:pressed {
                background-color: #B8D4FF; /* 按下时的更深蓝色 */
                color: black;           /* 按下时文字颜色 */
                padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                padding-top: 3px;
                border: 2px solid black; /* 按下时边框颜色 */
            }
        """)
        self.btn_hw_save1.setGeometry(965, 200, 200, 50)
        self.btn_hw_save1.clicked.connect(self.save_handwriting)

        # 创建识别结果标签
        self.label_result1 = QLabel(self.handwriting_interface)
        self.label_result1.setStyleSheet("""
                                    QLabel{
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                        border: 2px solid black; /* 边框属性 */
                                    }
                                """)
        self.label_result1.setGeometry(1000, 280, 150, 150)

        # 创建结果显示标签
        self.label_num1 = QLabel("识别结果:  准确度:  ", self.handwriting_interface)
        self.label_num1.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_num1.setStyleSheet("""
                    QLabel {
                        background-color: rgba(230, 240, 255, 0);  /* 标签背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                    }
                """)
        self.label_num1.setGeometry(865, 460, 400, 50)

        # 创建图片显示标签
        self.label_file2 = QLabel(self.file_interface)
        self.label_file2.setStyleSheet("""
                                    QLabel{
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                        border-image: url(./ui_img/image.jpg) 0 0 0 0 stretch stretch;  /* 背景图片 */
                                    }
                                """)
        self.label_file2.setGeometry(50, 5, 800, 520)

        # 创建上传文件按钮
        self.btn_file2 = QPushButton("上 传 文 件", self.file_interface)
        self.btn_file2.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                        border: 2px solid black; /* 边框属性 */
                    }
                    QPushButton:hover {
                        background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                        border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
                    }
                    QPushButton:pressed {
                        background-color: #B8D4FF; /* 按下时的更深蓝色 */
                        color: black;           /* 按下时文字颜色加深 */
                        padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                        padding-top: 3px;
                        border: 2px solid black; /* 按下时边框颜色 */
                    }
                """)
        self.btn_file2.setGeometry(965, 40, 200, 50)
        self.btn_file2.clicked.connect(self.open_file)

        # 创建识别数字按钮
        self.btn_file_recog2 = QPushButton("识 别 数 字", self.file_interface)
        self.btn_file_recog2.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                        border: 2px solid black; /* 边框属性 */
                    }
                    QPushButton:hover {
                        background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                        border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
                    }
                    QPushButton:pressed {
                        background-color: #B8D4FF; /* 按下时的更深蓝色 */
                        color: black;           /* 按下时文字颜色加深 */
                        padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                        padding-top: 3px;
                        border: 2px solid black; /* 按下时边框颜色 */
                    }
                """)
        self.btn_file_recog2.setGeometry(965, 120, 200, 50)
        self.btn_file_recog2.clicked.connect(self.recognize_file)

        # 创建清空图片按钮
        self.btn_flie_clear2 = QPushButton("清 空 图 片", self.file_interface)
        self.btn_flie_clear2.setStyleSheet("""
            QPushButton {
                background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                color: black;              /* 文字颜色 - 黑色 */
                border-radius: 5px;        /* 圆角半径 */
                font: bold 22px;           /* 字体加粗，22像素 */
                padding: 5px;              /* 内边距 */
                text-align: center;        /* 文字居中  */
                border: 2px solid black; /* 边框属性 */
            }
            QPushButton:hover {
                background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
            }
            QPushButton:pressed {
                background-color: #B8D4FF; /* 按下时的更深蓝色 */
                color: black;           /* 按下时文字颜色 */
                padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                padding-top: 3px;
                border: 2px solid black; /* 按下时边框颜色 */
            }
        """)
        self.btn_flie_clear2.setGeometry(965, 200, 200, 50)
        self.btn_flie_clear2.clicked.connect(self.clear_file)

        # 创建识别结果标签
        self.label_result2 = QLabel(self.file_interface)
        self.label_result2.setStyleSheet("""
                                    QLabel{
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                        border: 2px solid black; /* 边框属性 */
                                    }
                                """)
        self.label_result2.setGeometry(1000, 280, 150, 150)

        # 创建结果显示标签
        self.label_num2 = QLabel("识别结果:  准确度:  ", self.file_interface)
        self.label_num2.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_num2.setStyleSheet("""
                    QLabel {
                        background-color: rgba(230, 240, 255, 0);  /* 标签背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                    }
                """)
        self.label_num2.setGeometry(865, 460, 400, 50)

        # 创建摄像头画面显示标签
        self.label_camera3 = QLabel(self.camera_interface)
        self.label_camera3.setStyleSheet("""
                                    QLabel{
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                        border-image: url(./ui_img/image2.jpg) 0 0 0 0 stretch stretch;  /* 背景图片 */
                                    }
                                """)
        self.label_camera3.setGeometry(50, 5, 800, 520)

        # 创建开启摄像头按钮
        self.btn_open_camera3 = QPushButton("开 启 摄 像 头", self.camera_interface)
        self.btn_open_camera3.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                        border: 2px solid black; /* 边框属性 */
                    }
                    QPushButton:hover {
                        background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                        border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
                    }
                    QPushButton:pressed {
                        background-color: #B8D4FF; /* 按下时的更深蓝色 */
                        color: black;           /* 按下时文字颜色加深 */
                        padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                        padding-top: 3px;
                        border: 2px solid black; /* 按下时边框颜色 */
                    }
                """)
        self.btn_open_camera3.setGeometry(965, 40, 200, 50)
        self.btn_open_camera3.clicked.connect(self.open_camera)

        # 创建保存视频帧按钮
        self.btn_save_camera3 = QPushButton("保 存 视 频 帧", self.camera_interface)
        self.btn_save_camera3.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                        border: 2px solid black; /* 边框属性 */
                    }
                    QPushButton:hover {
                        background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                        border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
                    }
                    QPushButton:pressed {
                        background-color: #B8D4FF; /* 按下时的更深蓝色 */
                        color: black;           /* 按下时文字颜色加深 */
                        padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                        padding-top: 3px;
                        border: 2px solid black; /* 按下时边框颜色 */
                    }
                """)
        self.btn_save_camera3.setGeometry(965, 120, 200, 50)
        self.btn_save_camera3.clicked.connect(self.save_camera)

        # 创建关闭摄像头按钮
        self.btn_close_camera3 = QPushButton("关 闭 摄 像 头", self.camera_interface)
        self.btn_close_camera3.setStyleSheet("""
            QPushButton {
                background-color: rgba(230, 240, 255, 0);  /* 按钮背景色 */
                color: black;              /* 文字颜色 - 黑色 */
                border-radius: 5px;        /* 圆角半径 */
                font: bold 22px;           /* 字体加粗，22像素 */
                padding: 5px;              /* 内边距 */
                text-align: center;        /* 文字居中  */
                border: 2px solid black; /* 边框属性 */
            }
            QPushButton:hover {
                background-color: #D1E2FF; /* 鼠标悬停时的稍深蓝色 */
                border: 2px solid #7AA9FF; /* 悬停时边框颜色 */
            }
            QPushButton:pressed {
                background-color: #B8D4FF; /* 按下时的更深蓝色 */
                color: black;           /* 按下时文字颜色 */
                padding-left: 3px;        /* 按下效果 - 向右下偏移 */
                padding-top: 3px;
                border: 2px solid black; /* 按下时边框颜色 */
            }
        """)
        self.btn_close_camera3.setGeometry(965, 200, 200, 50)
        self.btn_close_camera3.clicked.connect(self.close_camera)

        # 创建识别结果标签
        self.label_result3 = QLabel(self.camera_interface)
        self.label_result3.setStyleSheet("""
                                    QLabel{
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                        border: 2px solid black; /* 边框属性 */
                                    }
                                """)
        self.label_result3.setGeometry(1000, 280, 150, 150)

        # 创建结果显示标签
        self.label_num3 = QLabel("识别结果:  准确度:  ", self.camera_interface)
        self.label_num3.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_num3.setStyleSheet("""
                    QLabel {
                        background-color: rgba(230, 240, 255, 0);  /* 标签背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                    }
                """)
        self.label_num3.setGeometry(865, 460, 400, 50)

    def addSubInterface(self, widget: QLabel, objectName, text):
        widget.setObjectName(objectName)
        widget.setAlignment(Qt.AlignCenter)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(routeKey=objectName, text=text)

    def recognize_handwriting(self):
        """识别手写板内容的槽函数（支持背景图像）"""
        try:
            # 创建QPixmap作为画布
            pixmap = QPixmap(self.handwriting.size())
            pixmap.fill(Qt.black)  # 黑色背景

            # 创建QPainter绘制手写笔迹
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

            # 复制手写板的绘制内容
            pen = QPen(self.handwriting.pen_color,
                       self.handwriting.pen_width,
                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)

            # 绘制所有线段（与HandwritingWidget.paintEvent保持一致）
            points = self.handwriting.points
            if len(points) > 1:
                for i in range(1, len(points)):
                    if points[i - 1] and points[i]:
                        painter.drawLine(points[i - 1], points[i])

            painter.end()  # 结束绘制

            # 生成保存路径（按时间戳命名）
            temp_dir = "./"
            temp_path = os.path.join(temp_dir, f"hw_temp.png")

            # 保存图片（PNG格式保留黑色背景）
            if pixmap.save(temp_path, "PNG"):
                print("successful！")
            else:
                print("error！")

            # OpenCV图像预处理（参考MNIST格式要求）
            img = cv2.imread(temp_path)  # 读取输入图像
            img_bw = g_n(img)  # 调用预处理函数，返回二值化后的图像（仅保留数字区域）

            # 计算每行和每列的白色像素数量
            img_bw_c = img_bw.sum(axis=1) / 255  # 每行的白色像素数（行方向总和 / 255）
            img_bw_r = img_bw.sum(axis=0) / 255  # 每列的白色像素数（列方向总和 / 255）

            r_ind, c_ind = [], []  # 初始化存储列和行索引的列表

            # 确定数字区域的列边界
            for k, r in enumerate(img_bw_r):  # 遍历每列的白色像素数
                if r >= 5:  # 如果该列的白色像素数≥5
                    r_ind.append(k)  # 记录该列索引

            # 确定数字区域的行边界
            for k, c in enumerate(img_bw_c):  # 遍历每行的白色像素数
                if c >= 5:  # 如果该行的白色像素数≥5
                    c_ind.append(k)  # 记录该行索引

            # 裁剪出包含数字的区域
            img_bw_sg = img_bw[c_ind[0]:c_ind[-1], r_ind[0]:r_ind[-1]]

            # 计算裁剪后的图像尺寸
            leng_c = len(c_ind)  # 裁剪后的行数（高度）
            leng_r = len(r_ind)  # 裁剪后的列数（宽度）

            # 设置目标边长为裁剪后的高度+20，用于调整为正方形
            side_len = leng_c + 20
            add_r = int((side_len - leng_r) / 2)  # 计算左右边框宽度

            # 添加边框使图像居中并调整为正方形
            img_bw_sg_bord = cv2.copyMakeBorder(
                img_bw_sg, 10, 10, add_r, add_r,  # 上下各10像素，左右各add_r像素
                cv2.BORDER_CONSTANT, value=[0, 0, 0]  # 黑色填充
            )

            # 调整图像尺寸为28x28（模型输入尺寸）
            img_in = cv2.resize(img_bw_sg_bord, (28, 28))

            # 使用模型预测并处理结果
            result_org = pt.predict(img_in, self.net)  # 获取原始预测结果（logits）
            result = softmax(result_org)  # 应用softmax获取概率分布
            best_result = result.argmax(dim=1).item()  # 找到概率最大的类别
            best_result_num = max(max(result)).cpu().detach().numpy()  # 提取最大概率值

            # 若概率≤0.5则认为结果不可靠
            if best_result_num <= 0.5:
                best_result = None

            # 显示结果
            show_image_on_label(img_bw_sg_bord, self.label_result1, 150, 150)  # 裁剪并添加边框后的图像
            self.createSuccessInfoBar("识别成功", f"识别结果:{best_result}，准确率:{best_result_num * 100:.2f}%")
            self.label_num1.setText(f"识别结果:{best_result}  准确度:{best_result_num * 100:.2f}%")

            # 打印结果
            # print(result)
            print("*" * 50)
            print("The number is:", best_result)
        except Exception as e:
            self.createErrorInfoBar("识别错误", f"发生错误：{str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def save_handwriting(self):
        """保存手写板内容为PNG图片"""
        try:
            # 创建QPixmap作为画布
            pixmap = QPixmap(self.handwriting.size())
            pixmap.fill(Qt.black)  # 黑色背景

            # 创建QPainter绘制到pixmap上
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

            # 复制手写板的绘制内容
            pen = QPen(self.handwriting.pen_color,
                       self.handwriting.pen_width,
                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)

            # 绘制所有线段（与HandwritingWidget.paintEvent保持一致）
            points = self.handwriting.points
            if len(points) > 1:
                for i in range(1, len(points)):
                    if points[i - 1] and points[i]:
                        painter.drawLine(points[i - 1], points[i])

            painter.end()  # 结束绘制

            # 生成保存路径（按时间戳命名）
            save_dir = "./save_img"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"hw_{timestamp}.png")

            # 保存图片（PNG格式保留黑色背景）
            if pixmap.save(save_path, "PNG"):
                self.createSuccessInfoBar("保存成功", f"字迹已保存至：{save_path}")
            else:
                self.createErrorInfoBar("保存失败", "无法保存图片，请检查路径权限")

        except Exception as e:
            self.createErrorInfoBar("错误", f"保存过程中发生错误：{str(e)}")

    def open_file(self):
        """打开图片并展示在标签中"""
        # 使用QFileDialog获取图片路径
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            try:
                # 加载图片
                image = cv2.imread(file_path)
                # 显示图片到标签
                show_image_on_label(image, self.label_file2, 800, 520)
                # 调用成功提示框
                self.createSuccessInfoBar("图片加载成功", f"已加载图片: {file_path}")
                # 可选：保存当前图片路径供后续使用
                self.current_image_path = file_path
                print(self.current_image_path)
            except Exception as e:
                self.createErrorInfoBar("图片加载失败", str(e))

    def recognize_file(self):
        """识别手写板内容的槽函数（支持背景图像）"""
        try:
            img = cv2.imread(self.current_image_path)  # 读取输入图像
            img_bw = g_n_2(img)  # 调用预处理函数，返回二值化后的图像（仅保留数字区域）
            # 计算每行和每列的白色像素数量
            img_bw_c = img_bw.sum(axis=1) / 255  # 每行的白色像素数（行方向总和 / 255）
            img_bw_r = img_bw.sum(axis=0) / 255  # 每列的白色像素数（列方向总和 / 255）

            r_ind, c_ind = [], []  # 初始化存储列和行索引的列表

            # 确定数字区域的列边界
            for k, r in enumerate(img_bw_r):  # 遍历每列的白色像素数
                if r >= 5:  # 如果该列的白色像素数≥5
                    r_ind.append(k)  # 记录该列索引

            # 确定数字区域的行边界
            for k, c in enumerate(img_bw_c):  # 遍历每行的白色像素数
                if c >= 5:  # 如果该行的白色像素数≥5
                    c_ind.append(k)  # 记录该行索引

            # 裁剪出包含数字的区域
            img_bw_sg = img_bw[c_ind[0]:c_ind[-1], r_ind[0]:r_ind[-1]]

            # 计算裁剪后的图像尺寸
            leng_c = len(c_ind)  # 裁剪后的行数（高度）
            leng_r = len(r_ind)  # 裁剪后的列数（宽度）

            # 设置目标边长为裁剪后的高度+20，用于调整为正方形
            side_len = leng_c + 20
            add_r = int((side_len - leng_r) / 2)  # 计算左右边框宽度

            # 添加边框使图像居中并调整为正方形
            img_bw_sg_bord = cv2.copyMakeBorder(
                img_bw_sg, 10, 10, add_r, add_r,  # 上下各10像素，左右各add_r像素
                cv2.BORDER_CONSTANT, value=[0, 0, 0]  # 黑色填充
            )

            # 调整图像尺寸为28x28（模型输入尺寸）
            img_in = cv2.resize(img_bw_sg_bord, (28, 28))

            # 使用模型预测并处理结果
            result_org = pt.predict(img_in, self.net)  # 获取原始预测结果（logits）
            result = softmax(result_org)  # 应用softmax获取概率分布
            best_result = result.argmax(dim=1).item()  # 找到概率最大的类别
            best_result_num = max(max(result)).cpu().detach().numpy()  # 提取最大概率值

            # 若概率≤0.5则认为结果不可靠
            if best_result_num <= 0.5:
                best_result = None

            # 显示结果
            show_image_on_label(img_bw_sg_bord, self.label_result2, 150, 150)  # 裁剪并添加边框后的图像
            self.createSuccessInfoBar("识别成功", f"识别结果:{best_result}，准确率:{best_result_num * 100:.2f}%")
            self.label_num2.setText(f"识别结果:{best_result}  准确度:{best_result_num * 100:.2f}%")

            # 打印结果
            # print(result)
            print("*" * 50)
            print("The number is:", best_result)
        except Exception as e:
            self.createErrorInfoBar("识别错误", f"发生错误：{str(e)}")

    def clear_file(self):
        """清空标签控件中的图片"""
        try:
            # 检查标签控件是否存在
            if hasattr(self, 'label_file2') and self.label_file2 is not None:
                # 清空图片显示
                self.label_file2.clear()
                self.label_result2.clear()
                self.label_num2.setText("识别结果:  准确度:  ")
                # 清空保存的图片路径
                if hasattr(self, 'current_image_path'):
                    self.current_image_path = None
                # 显示操作成功的提示
                self.createSuccessInfoBar("操作成功", "已清空当前显示的图片")
            else:
                self.createWarningInfoBar("操作提示", "没有可清空的图片")
        except Exception as e:
            self.createErrorInfoBar("清空图片失败", str(e))

    def open_camera(self):
        """开启摄像头并显示画面"""
        try:
            # 初始化摄像头
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.createWarningInfoBar("操作提示", "摄像头打开失败，请检查设备连接")
            # 设置摄像头参数
            self.cap.set(3, 1920)
            self.cap.set(4, 1080)
            # 切换标志位状态
            self.camera_running = True
            self.createSuccessInfoBar("操作成功", "摄像头已开启")
            # 视频处理循环
            while self.camera_running:
                ret, frame = self.cap.read()
                self.current_frame = frame
                if not ret:
                    self.createWarningInfoBar("视频警告", "视频帧读取失败")
                    break
                try:
                    # 图像预处理
                    img_bw = g_n_2(frame)
                    img_bw_sg = get_roi(img_bw)
                    # 显示处理后的图像
                    show_image_on_label(img_bw_sg, self.label_result3, 150, 150)
                    # 数字识别
                    img_in = cv2.resize(img_bw_sg, (28, 28))
                    result_org = pt.predict(img_in, self.net)
                    result = softmax(result_org)
                    best_result = result.argmax(dim=1).item()
                    best_result_num = max(max(result)).cpu().detach().numpy()
                    if best_result_num <= 0.5:
                        best_result = None
                    # print(result)
                    print("*" * 50)
                    print("The number is:", best_result)
                    # 显示识别结果
                    show_image_on_label(frame, self.label_camera3, 800, 520)
                    self.label_num3.setText(f"识别结果:{best_result}  准确度:{best_result_num * 100:.2f}%")
                    # 处理系统事件
                    QApplication.processEvents()

                except Exception as process_error:
                    self.createErrorInfoBar("处理错误", f"图像处理失败: {str(process_error)}")
                    break

        except Exception as init_error:
            self.createErrorInfoBar("初始化错误", f"摄像头初始化失败: {str(init_error)}")
            self.camera_running = False
        finally:
            # 确保资源释放
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()

    def save_camera(self):
        """保存视频帧到./save_img目录，以时间戳命名"""
        try:
            # 确保保存目录存在
            save_dir = "./save_img"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 生成时间戳文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            # 检查是否有可保存的视频帧
            if not hasattr(self, 'current_frame') or self.current_frame is None:
                self.createWarningInfoBar("操作失败", "没有可用的视频帧")
                return
            # 保存帧
            success = cv2.imwrite(save_path, self.current_frame)
            if success:
                self.createSuccessInfoBar("操作成功", f"视频帧已保存到{save_path}")
            else:
                self.createErrorInfoBar("操作失败", "无法保存视频帧")
        except PermissionError:
            self.createErrorInfoBar("操作失败", "没有写入权限")
        except Exception as e:
            self.createErrorInfoBar("操作失败", f"发生错误：{str(e)}")

    def close_camera(self):
        """关闭摄像头并释放资源"""
        try:
            # 检查摄像头是否已打开
            if self.camera_running:
                # 切换标志位状态
                self.camera_running = False
                # 释放摄像头资源
                self.cap.release()
                # 清除残留属性
                del self.cap
                show_image_on_label(cv2.imread('./ui_img/image2.jpg'), self.label_camera3, 800, 520)
                self.label_result3.clear()
                self.label_num3.setText("识别结果:  准确度:  ")
                # 显示成功提示
                self.createSuccessInfoBar("操作成功", "摄像头已关闭")
            else:
                self.createWarningInfoBar("操作提示", "摄像头未开启或已关闭")
        except Exception as e:
            # 捕获异常并显示错误提示
            self.createErrorInfoBar("关闭摄像头失败", f"错误详情: {str(e)}")

    def onExitButtonClicked(self):
        """安全退出程序的槽函数"""
        # 检查并关闭摄像头
        if hasattr(self, 'cap') and self.cap.isOpened():
            print("正在关闭摄像头...")
            self.cap.release()
            print("摄像头已关闭")

        # 关闭Qt界面
        print("退出应用程序")
        self.close()

    def createSuccessInfoBar(self, title, content):
        """创建成功提示框"""
        InfoBar.success(
            title=title,  # 提示框标题
            content=content,  # 提示框内容
            orient=Qt.Horizontal,  # 水平布局
            isClosable=True,  # 允许用户手动关闭
            position=InfoBarPosition.TOP,  # 显示在窗口顶部
            duration=2000,  # 持续时间 2 秒
            parent=self  # 父组件为当前窗口
        )

    def createWarningInfoBar(self, title, content):
        """创建警告提示框"""
        InfoBar.warning(
            title=title,  # 提示框标题
            content=content,  # 提示框内容
            orient=Qt.Horizontal,  # 水平布局
            isClosable=True,  # 允许用户手动关闭
            position=InfoBarPosition.TOP,  # 显示在窗口顶部
            duration=2000,  # 持续时间 2 秒
            parent=self  # 父组件为当前窗口
        )

    def createErrorInfoBar(self, title, content):
        """创建失败提示框"""
        InfoBar.error(
            title=title,  # 提示框标题
            content=content,  # 提示框内容
            orient=Qt.Horizontal,  # 水平布局
            isClosable=True,  # 允许用户手动关闭
            position=InfoBarPosition.TOP,  # 显示在窗口顶部
            duration=2000,  # 持续时间 2 秒
            parent=self  # 父组件为当前窗口
        )

    # 定义额外的槽函数
    def on_interface_changed(self, widget):
        """当界面切换时执行的额外逻辑"""
        self.handwriting.clear()
        self.label_result1.clear()
        self.label_num1.setText("识别结果:  准确度:  ")
        self.label_file2.clear()
        self.label_result2.clear()
        self.label_num2.setText("识别结果:  准确度:  ")
        if self.camera_running:
            # 切换标志位状态
            self.camera_running = False
            # 释放摄像头资源
            self.cap.release()
            # 清除残留属性
            del self.cap
            show_image_on_label(cv2.imread('./ui_img/image2.jpg'), self.label_camera3, 800, 520)
            self.label_result3.clear()
            self.label_num3.setText("识别结果:  准确度:  ")
        if widget == self.handwriting_interface:
            # 在这里添加手写板界面特有的逻辑
            self.createSuccessInfoBar("切换成功", "切换到手写板界面")
        elif widget == self.camera_interface:
            # 在这里添加摄像头界面特有的逻辑
            self.createSuccessInfoBar("切换成功", "切换到摄像头界面")
        elif widget == self.file_interface:
            # 在这里添加文件界面特有的逻辑
            self.createSuccessInfoBar("切换成功", "切换到本地文件界面")

    def on_stack_changed(self, index):
        """当堆栈界面索引变化时执行的逻辑"""
        print(f"当前界面索引已更改为：{index}")
        current_widget = self.stackedWidget.widget(index)
        if current_widget:
            print(f"当前界面对象名：{current_widget.objectName()}")


class HandwritingWidget(QWidget):
    def __init__(self, parent=None):
        super(HandwritingWidget, self).__init__(parent)
        self.setMinimumSize(400, 300)

        # 初始化画笔设置
        self.pen_color = QColor(Qt.white)  # 白色笔迹
        self.pen_width = 7  # 笔迹宽度

        # 存储绘制点
        self.points = []

        # 设置黑色背景
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(palette)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

        # 设置画笔
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        # 绘制所有线段
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                if self.points[i - 1] is not None and self.points[i] is not None:
                    painter.drawLine(self.points[i - 1], self.points[i])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.points.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(None)  # 添加None表示一段连续笔画的结束

    def clear(self):
        """清除画板内容"""
        self.points = []
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Window = Window()
    Window.show()
    sys.exit(app.exec_())
