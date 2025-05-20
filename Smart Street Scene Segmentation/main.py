from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import *
from qfluentwidgets import InfoBar, InfoBarPosition
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox
from PyQt5 import uic
from tqdm import tqdm
import network
import utils
import os
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
import torch
import torch.nn as nn
from PIL import Image
from glob import glob

# 全局变量用于存储选择的图像路径
file_path = ""
# 全局变量用于存储分割后的图像路径
result_image_path = ""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 直接加载.ui文件
        uic.loadUi('GUI.ui', self)
        # 设置全屏显示
        # self.showFullScreen()
        # 设置窗口标题
        self.setWindowTitle("智慧街景分割")
        # 隐藏窗口标题栏
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 设置窗口图标
        self.setWindowIcon(QIcon("./source/icon.png"))
        # 禁用最大化按钮和双击标题栏最大化
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # 设置窗口大小（固定大小）
        self.setFixedSize(1280, 720)
        # 设置背景图像
        self.setStyleSheet("""
            QMainWindow {
                border-image: url('./source/background.jpg') 0 0 0 0 stretch stretch;
            }
        """)
        # 初始化
        self.initialize()

    def initialize(self):
        # 退出
        self.btn_exit.clicked.connect(self.close)
        # 模型选择
        options = ["模 型 选 择", "deeplabv3plus_mobilenet_cityscapes", "deeplabv3plus_mobilenet_voc"]
        self.ComboBox.addItems(options)
        # 载入图像
        self.btn_load.clicked.connect(self.load_image)
        # 语义分割
        self.btn_run.clicked.connect(self.segement_image)
        # 保存图像
        self.btn_save.clicked.connect(self.save_result_image)

    def load_image(self):
        # 打开文件选择对话框，只允许选择图像文件
        file_dialog = QFileDialog()
        global file_path
        file_path, _ = file_dialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg)")
        if file_path:
            self.createSuccessInfoBar("成功", f"已选择图像：{file_path}")
            print(f"选择的图像路径：{file_path}")
            # 显示图像
            pixmap = QPixmap(file_path)
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))

    def segement_image(self):
        global file_path
        global result_image_path
        # 判断图像路径是否为空
        if not file_path:
            self.createErrorInfoBar("错误", "未选择图像！")
            return
        # 获取下拉框中选择的内容
        selected_model = self.ComboBox.currentText()
        if selected_model == "模 型 选 择":
            self.createWarningInfoBar("警告", "请选择有效的模型！")
            return
        # 语义分割
        if selected_model == "deeplabv3plus_mobilenet_cityscapes":
            self.ckpt = './checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
            self.dataset = 'cityscapes'
        if selected_model == "deeplabv3plus_mobilenet_voc":
            self.ckpt = './checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth'
            self.dataset = 'voc'
        print(f"选择的模型：{self.ckpt}")
        # Datset Options
        self.input = file_path

        # Deeplab Options
        available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                                  not (name.startswith("__") or name.startswith('_')) and callable(
            network.modeling.__dict__[name])
                                  )
        self.model = 'deeplabv3plus_mobilenet'
        self.separable_conv = False
        self.output_stride = 16

        # Train Options
        self.save_val_results_to = 'test_results'
        self.crop_val = False
        self.val_batch_size = 4
        self.crop_size = 513
        self.gpu_id = '0'

        if self.dataset.lower() == 'voc':
            self.num_classes = 21
            decode_fn = VOCSegmentation.decode_target
        elif self.dataset.lower() == 'cityscapes':
            self.num_classes = 19
            decode_fn = Cityscapes.decode_target

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        # Setup dataloader
        image_files = []
        if os.path.isdir(self.input):
            for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
                files = glob(os.path.join(self.input, '**/*.%s' % (ext)), recursive=True)
                if len(files) > 0:
                    image_files.extend(files)
        elif os.path.isfile(self.input):
            image_files.append(self.input)

        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[self.model](num_classes=self.num_classes, output_stride=self.output_stride)
        if self.separable_conv and 'plus' in self.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)

        if self.ckpt is not None and os.path.isfile(self.ckpt):
            checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            print("Resume model from %s" % self.ckpt)
            del checkpoint
        else:
            print("[!] Retrain")
            model = nn.DataParallel(model)
            model.to(device)

        if self.crop_val:
            transform = T.Compose([
                T.Resize(self.crop_size),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        if self.save_val_results_to is not None:
            os.makedirs(self.save_val_results_to, exist_ok=True)
        with torch.no_grad():
            model = model.eval()
            for img_path in tqdm(image_files):
                ext = os.path.basename(img_path).split('.')[-1]
                img_name = os.path.basename(img_path)[:-len(ext) - 1]
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0)  # To tensor of NCHW
                img = img.to(device)

                pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
                colorized_preds = decode_fn(pred).astype('uint8')
                colorized_preds = Image.fromarray(colorized_preds)
                if self.save_val_results_to:
                    result_image_path = os.path.join(self.save_val_results_to, img_name + '.png')
                    colorized_preds.save(result_image_path)
        # 显示分割结果图片
        if result_image_path:
            pixmap = QPixmap(result_image_path)
            self.label_2.setAlignment(Qt.AlignCenter)
            self.label_2.setPixmap(
                pixmap.scaled(self.label_2.width(), self.label_2.height(), Qt.KeepAspectRatio))
            self.createSuccessInfoBar("成功", "语义分割完成，结果已显示。")

    def save_result_image(self):
        global result_image_path
        if not result_image_path:
            self.createErrorInfoBar("警告", "没有分割结果图片可供保存！")
            return
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "保存分割结果图片", "", "PNG 图像 (*.png)")
        if save_path:
            try:
                image = Image.open(result_image_path)
                image.save(save_path)
                self.createSuccessInfoBar("成功", f"分割结果图片已保存到：{save_path}")
            except Exception as e:
                self.createErrorInfoBar("错误", f"保存图片时出现错误：{str(e)}")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
