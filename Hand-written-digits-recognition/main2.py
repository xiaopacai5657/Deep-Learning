from PyQt5.QtWidgets import *
import time
import os
from Pre_treatment import get_number as g_n
import predict as pt
from Pre_treatment import softmax
from gui_util import *
from qfluentwidgets import InfoBar, InfoBarPosition
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel


# 手写数字识别
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.btn_exit = None  # 退出系统按钮
        self.btn_hw_file = None  # 上传文件
        self.btn_hw_recog = None  # 识别数字
        self.btn_hw_clear = None  # 清空写字板
        self.btn_hw_save = None  # 保存字迹
        self.label_result = None  # 识别结果标签
        self.label_num = None  # 结果显示标签
        self.handwriting = None  # 写字板
        self.statusbar = None  # 状态栏
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

        # 创建手写板控件
        self.handwriting = HandwritingWidget(self)
        self.handwriting.setGeometry(50, 150, 800, 520)

        # 创建上传文件按钮
        self.btn_hw_file = QPushButton(" 上 传 文 件 ", self)
        self.btn_hw_file.setStyleSheet("""
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
        self.btn_hw_file.setGeometry(965, 160, 200, 50)
        self.btn_hw_file.clicked.connect(self.handwriting.open_image)

        # 创建识别数字按钮
        self.btn_hw_recog = QPushButton(" 识 别 数 字 ", self)
        self.btn_hw_recog.setStyleSheet("""
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
        self.btn_hw_recog.setGeometry(965, 230, 200, 50)
        self.btn_hw_recog.clicked.connect(self.recognize_handwriting)

        # 创建清空写字板按钮
        self.btn_hw_clear = QPushButton("清 空 写 字 板", self)
        self.btn_hw_clear.setStyleSheet("""
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
        self.btn_hw_clear.setGeometry(965, 300, 200, 50)
        self.btn_hw_clear.clicked.connect(self.handwriting.clear)

        # 创建保存字迹按钮
        self.btn_hw_save = QPushButton(" 保 存 字 迹 ", self)
        self.btn_hw_save.setStyleSheet("""
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
        self.btn_hw_save.setGeometry(965, 370, 200, 50)
        self.btn_hw_save.clicked.connect(self.save_handwriting)

        # 创建识别结果标签
        self.label_result = QLabel(self)
        self.label_result.setStyleSheet("""
                                    QLabel{
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                        border: 2px solid black; /* 边框属性 */
                                    }
                                """)
        self.label_result.setGeometry(1000, 440, 150, 150)

        # 创建结果显示标签
        self.label_num = QLabel("识 别 结 果：  准 确 度： ", self)
        self.label_num.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_num.setStyleSheet("""
                    QLabel {
                        background-color: rgba(230, 240, 255, 0);  /* 标签背景色 */
                        color: black;              /* 文字颜色 - 黑色 */
                        border-radius: 5px;        /* 圆角半径 */
                        font: bold 22px;           /* 字体加粗，22像素 */
                        padding: 5px;              /* 内边距 */
                        text-align: center;        /* 文字居中  */
                        border: 2px solid black; /* 边框属性 */
                    }
                """)
        self.label_num.setGeometry(865, 610, 400, 50)

    def recognize_handwriting(self):
        """识别手写板内容的槽函数（支持背景图像）"""
        try:
            net = pt.get_net()  # 加载预训练的神经网络模型

            # 创建QPixmap作为画布
            pixmap = QPixmap(self.handwriting.size())
            if self.handwriting.background_image:
                # 如果有背景图，先绘制背景
                painter = QPainter(pixmap)
                painter.drawImage(self.handwriting.rect(),
                                  self.handwriting.background_image)
                painter.end()
            else:
                pixmap.fill(Qt.black)  # 无背景时填充黑色

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
            result_org = pt.predict(img_in, net)  # 获取原始预测结果（logits）
            result = softmax(result_org)  # 应用softmax获取概率分布
            best_result = result.argmax(dim=1).item()  # 找到概率最大的类别
            best_result_num = max(max(result)).cpu().detach().numpy()  # 提取最大概率值

            # 若概率≤0.5则认为结果不可靠
            if best_result_num <= 0.5:
                best_result = None

            # 显示结果
            show_image_on_label(img_bw_sg_bord, self.label_result, 150, 150)  # 裁剪并添加边框后的图像
            self.createSuccessInfoBar("识别成功", f"识别结果：{best_result}，准确率：{best_result_num * 100:.2f}%")
            self.label_num.setText(f"识 别 结 果：{best_result} 准 确 度：{best_result_num * 100:.2f}%")

            # 打印结果
            print(result)
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
            save_dir = "save_img"
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


class HandwritingWidget(QWidget):
    def __init__(self, parent=None):
        super(HandwritingWidget, self).__init__(parent)
        self.setMinimumSize(400, 300)

        # 初始化画笔设置
        self.pen_color = QColor(Qt.white)  # 白色笔迹
        self.pen_width = 7  # 笔迹宽度

        # 存储绘制点
        self.points = []

        # 存储背景图片
        self.background_image = None

        # 设置黑色背景
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(palette)

        # self.setAutoFillBackground(True)
        # palette = self.palette()
        # blackboard_color = QColor(92, 108, 98)
        # palette.setColor(self.backgroundRole(), blackboard_color)
        # self.setPalette(palette)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿

        # 如果有背景图片，先绘制背景图片
        if self.background_image:
            painter.drawImage(self.rect(), self.background_image)

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
        self.background_image = None
        self.update()
        # 调用成功提示框
        self.parent().createSuccessInfoBar("清除成功", "写字板内容已清空")

    def open_image(self):
        """打开图片并设置为背景"""
        # 使用QFileDialog获取图片路径
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if file_path:
            # 加载图片并缩放以适应画板大小
            image = QImage(file_path)
            if not image.isNull():
                self.background_image = image.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.update()
                # 调用成功提示框
                self.parent().createSuccessInfoBar("图片加载成功", f"已加载图片: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Window = Window()
    Window.show()
    sys.exit(app.exec_())
