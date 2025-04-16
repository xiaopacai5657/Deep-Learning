import cv2
import mediapipe as mp
import time
import math
from PyQt5.QtWidgets import *
import time
import os
from gui_util import *
from qfluentwidgets import InfoBar, InfoBarPosition
import sys
import threading
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QStackedWidget, QLabel
from qfluentwidgets import SegmentedWidget

# 定义关键的全局变量
frame = None


# 趣味虚拟拖拽
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.infoBarHelper = InfoBarHelper()  # 初始化工具类实例
        self.btn_exit = None  # 退出系统按钮
        self.label_camera = None  # 摄像头画面显示标签
        self.timer = None  # 定时器
        self.n = 0  # 初始信号
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
                border-image: url('./ui_img/background.jpg') 0 0 0 0 stretch stretch;
            }
        """)
        # 加载setup_ui函数
        self.setup_ui()

    def setup_ui(self):
        # 创建退出系统按钮
        self.btn_exit = QPushButton(self)  # 添加按钮文本
        # 设置退出系统按钮的样式
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0);  /* 初始状态 - 全透明 */
                border: none;                              /* 无边框 */
                border-radius: 5px;                        /* 圆角半径 */
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 100); /* 鼠标悬停 - 半透明浅蓝色 */
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 125); /* 按下时 - 更深的半透明蓝色 */
                padding-left: 3px;                          /* 按下效果 - 向右下偏移 */
                padding-top: 3px;
            }
        """)
        # 设置按钮的位置和大小 setGeometry(x, y, width, height)
        self.btn_exit.setGeometry(1140, 45, 107, 40)
        # 连接按钮的点击信号到槽函数
        self.btn_exit.clicked.connect(self.onExitButtonClicked)

        # 创建摄像头画面显示标签
        self.label_camera = QLabel(self)
        self.label_camera.setStyleSheet("""
                                    QLabel{
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                        border-image: url(./ui_img/image.jpg) 0 0 0 0 stretch stretch;  /* 背景图片 */
                                    }
                                """)
        self.label_camera.setGeometry(25, 125, 1230, 575)

        # 创建定时器
        self.timer = QTimer(self)
        # 连接定时器的 timeout 信号到 self.tick_callback 方法
        self.timer.timeout.connect(self.tick_callback)
        # 设置定时器间隔为 30 毫秒并启动定时器
        self.timer.start(30)

    def tick_callback(self):
        if self.n == 0:
            self.n += 1
            self.infoBarHelper.show_success(self, '正在初始化', '请稍后……')
            # 创建另一线程t2并运行，可以保证程序运行流畅
            t2 = threading.Thread(target=self.detect)
            # 创建线程守护
            t2.setDaemon(True)
            # 启动子线程
            t2.start()
        else:
            # 添加帧有效性校验逻辑
            if self.is_valid_frame(frame):
                # 调用显示函数前再次验证帧数据
                show_image_on_label(frame, self.label_camera, 1280, 720)

    def detect(self):
        """检测手势的槽函数"""
        control = HandControlVolume()
        control.recognize()

    def is_valid_frame(self, frame):
        """ 验证视频帧有效性的方法 """
        # 有效性判断条件（根据实际使用的视频流框架调整）
        return (
                frame is not None and  # 非空对象
                hasattr(frame, 'shape') and  # 有形状属性（OpenCV帧特征）
                len(frame.shape) >= 2 and  # 至少二维数组（宽高）
                frame.size > 0 and  # 非空数据
                frame.shape[0] > 0 and frame.shape[1] > 0  # 宽高大于零
        )

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


# 创建方块管理类
class SquareManager:
    def __init__(self, rect_width):
        # 方块的边长
        self.rect_width = rect_width

        # 存储所有方块左上角的横纵坐标和透明度
        self.square_count = 0
        self.rect_left_x_list = []
        self.rect_left_y_list = []
        self.alpha_list = []

        # 记录手指与方块左上角的距离（用于拖动）
        self.L1 = 0
        self.L2 = 0

        # 拖动模式激活标志
        self.drag_active = False

        # 当前被激活（选中）的方块索引
        self.active_index = -1

    # 创建一个新的方块（不立即显示）
    def create(self, rect_left_x, rect_left_y, alpha=0.4):
        self.rect_left_x_list.append(rect_left_x)
        self.rect_left_y_list.append(rect_left_y)
        self.alpha_list.append(alpha)
        self.square_count += 1

    # 显示所有方块（根据是否被激活显示不同颜色）
    def display(self, class_obj):
        for i in range(self.square_count):
            x = self.rect_left_x_list[i]
            y = self.rect_left_y_list[i]
            alpha = self.alpha_list[i]

            overlay = class_obj.image.copy()

            if i == self.active_index:
                # 被激活的方块为紫色
                cv2.rectangle(overlay, (x, y), (x + self.rect_width, y + self.rect_width), (255, 0, 255), -1)
            else:
                # 其他方块为蓝色
                cv2.rectangle(overlay, (x, y), (x + self.rect_width, y + self.rect_width), (255, 0, 0), -1)

            # 将透明方块叠加到图像上
            class_obj.image = cv2.addWeighted(overlay, alpha, class_obj.image, 1 - alpha, 0)

    # 检查坐标点是否在某个方块内，返回该方块的索引
    def checkOverlay(self, check_x, check_y):
        for i in range(self.square_count):
            x = self.rect_left_x_list[i]
            y = self.rect_left_y_list[i]

            if x < check_x < (x + self.rect_width) and y < check_y < (y + self.rect_width):
                self.active_index = i  # 设置当前激活的方块索引
                return i

        return -1  # 没有方块被命中

    # 设置手指与被激活方块左上角的相对距离（用于拖动计算）
    def setLen(self, check_x, check_y):
        self.L1 = check_x - self.rect_left_x_list[self.active_index]
        self.L2 = check_y - self.rect_left_y_list[self.active_index]

    # 根据新的指尖位置更新被激活方块的位置
    def updateSquare(self, new_x, new_y):
        self.rect_left_x_list[self.active_index] = new_x - self.L1
        self.rect_left_y_list[self.active_index] = new_y - self.L2


# 识别控制类
class HandControlVolume:
    def __init__(self):
        # 初始化 mediapipe 模块
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # 中指与矩形左上角点的距离（用于拖动计算）
        self.L1 = 0
        self.L2 = 0

        # 当前帧图像（供方块管理器调用）
        self.image = None

    # 主函数：进行手势识别和交互控制
    def recognize(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 获取摄像头画面分辨率
        resize_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resize_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 显示指尖距离文本
        rect_percent_text = 0

        # 初始化方块管理器，并创建方块
        squareManager = SquareManager(150)
        for i in range(2):
            squareManager.create(200 * i + 20, 200, 0.6)

        # 初始化手部检测模块
        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:

            while cap.isOpened():
                success, self.image = cap.read()
                self.image = cv2.resize(self.image, (resize_w, resize_h))

                if not success:
                    print("摄像头初始化失败")
                    continue

                # 图像预处理
                self.image.flags.writeable = False
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image = cv2.flip(self.image, 1)
                results = hands.process(self.image)
                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                # 如果检测到手
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 绘制手部关键点
                        self.mp_drawing.draw_landmarks(
                            self.image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        landmark_list = []
                        paw_x_list = []
                        paw_y_list = []

                        # 获取所有关键点
                        for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                            landmark_list.append([landmark_id, finger_axis.x, finger_axis.y, finger_axis.z])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)

                        if landmark_list:
                            # 定义像素转换函数
                            to_px_x = lambda x: math.ceil(x * resize_w)
                            to_px_y = lambda y: math.ceil(y * resize_h)

                            # 画手掌边框
                            x1, x2 = map(to_px_x, [min(paw_x_list), max(paw_x_list)])
                            y1, y2 = map(to_px_y, [min(paw_y_list), max(paw_y_list)])
                            cv2.rectangle(self.image, (x1 - 30, y1 - 30), (x2 + 30, y2 + 30), (0, 255, 0), 2)

                            # 中指和食指指尖位置
                            mid_tip = landmark_list[12]
                            idx_tip = landmark_list[8]

                            mid_point = (to_px_x(mid_tip[1]), to_px_y(mid_tip[2]))
                            idx_point = (to_px_x(idx_tip[1]), to_px_y(idx_tip[2]))
                            mid_idx_center = ((mid_point[0] + idx_point[0]) // 2,
                                              (mid_point[1] + idx_point[1]) // 2)

                            # 绘制两个指尖和中心点
                            draw_circle = lambda pt: cv2.circle(self.image, pt, 10, (255, 0, 255), -1)
                            for pt in [mid_point, idx_point, mid_idx_center]:
                                self.image = draw_circle(pt)

                            # 绘制连接线
                            self.image = cv2.line(self.image, mid_point, idx_point, (255, 0, 255), 5)

                            # 计算距离
                            line_len = math.hypot(idx_point[0] - mid_point[0],
                                                  idx_point[1] - mid_point[1])
                            rect_percent_text = math.ceil(line_len)

                            # 拖动方块逻辑
                            if squareManager.drag_active:
                                squareManager.updateSquare(mid_idx_center[0], mid_idx_center[1])
                                if line_len > 60:
                                    squareManager.drag_active = False
                                    squareManager.active_index = -1
                            elif (line_len < 60 and
                                  squareManager.checkOverlay(mid_idx_center[0], mid_idx_center[1]) != -1 and
                                  not squareManager.drag_active):
                                squareManager.drag_active = True
                                squareManager.setLen(mid_idx_center[0], mid_idx_center[1])

                # 显示方块
                squareManager.display(self)

                # # 画文字信息
                # cv2.putText(self.image, f"Distance: {rect_percent_text}", (10, 120),
                #             cv2.FONT_HERSHEY_TRIPLEX , 3, (255, 0, 0), 3)

                cv2.putText(self.image,
                            f"Active: {'None' if squareManager.active_index == -1 else squareManager.active_index}",
                            (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (51, 51, 204), 2)  # 柔和的红色字体
                # 绘制方框
                cv2.rectangle(self.image, (1000, 200), (1200, 400), (51, 51, 204), 2)

                # 包含换行符的文本
                text = "Placement\n   Area"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                color = (51, 51, 204)
                thickness = 2

                # 拆分文本为多行
                lines = text.split('\n')
                line_height = cv2.getTextSize('A', font, font_scale, thickness)[0][1] + 10  # 行高，加 10 是为了增加间距

                # 计算文字整体的宽度和高度
                total_text_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines])
                total_text_height = len(lines) * line_height

                # 计算文字的 x 坐标，使其在方框内水平居中
                text_x = 1000 + (1200 - 1000 - total_text_width) // 2
                # 计算文字的起始 y 坐标，使其在方框内垂直居中
                text_y = 200 + (400 - 200 - total_text_height) // 2 + line_height

                # 逐行绘制文字
                for line in lines:
                    cv2.putText(self.image, line, (text_x, text_y), font, font_scale, color, thickness)
                    text_y += line_height

                global frame
                frame = self.image


# 新增提示框工具类
class InfoBarHelper:
    """提示框工具类，提供标准化提示组件"""

    @staticmethod
    def show_success(parent, title: str, content: str):
        """成功提示框"""
        InfoBar.success(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=parent
        )

    @staticmethod
    def show_warning(parent, title: str, content: str):
        """警告提示框"""
        InfoBar.warning(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=parent
        )

    @staticmethod
    def show_error(parent, title: str, content: str):
        """错误提示框"""
        InfoBar.error(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=parent
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Window = Window()
    Window.show()
    sys.exit(app.exec_())
