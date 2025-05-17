from gui_util import *
import threading
import os
import sys
from pathlib import Path
import time
import torch
from PyQt5.Qt import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.btn_close = None  # 关闭系统按钮
        self.label_video = None  # 实时视频显示标签
        self.label_smoking = None  # 吸烟显示标签
        self.label_motorcycle = None  # 电动车显示标签
        self.label_bicycle = None  # 自行车显示标签
        self.label_person = None  # 人员数量显示标签
        self.label_result = None  # 结果显示标签
        self.smoking = 0  # 吸烟
        self.motorcycle = 0  # 电动车
        self.bicycle = 0  # 自行车
        self.person = 0  # 人员数量
        self.timer = None
        self.im1 = cv2.imread('ui_img/lift.png', cv2.IMREAD_COLOR)
        self.result = cv2.imread('ui_img/wxts.png', cv2.IMREAD_COLOR)
        self.n = 0
        self.p = None
        self.cap = cv2.VideoCapture()
        # 设置全屏显示
        # self.showFullScreen()
        # 设置窗口标题
        self.setWindowTitle("智慧电梯行为分析系统")
        # 隐藏窗口标题栏
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 设置窗口图标
        self.setWindowIcon(QIcon("./ui_img/icon.png"))
        # 禁用最大化按钮和双击标题栏最大化
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # 设置窗口背景图并重置图片的大小,参数分别为高度、宽度,不保持纵横比
        pixmap = QPixmap("./ui_img/background.jpg").scaled(1280, 720, Qt.IgnoreAspectRatio)
        palette = QPalette()  # 创建一个新的调色板对象
        palette.setBrush(QPalette.Background, QBrush(pixmap))  # 设置调色板的背景画刷为使用pixmap的画刷，即设置背景图
        self.setPalette(palette)  # 将这个调色板应用到window窗口上
        # 设置窗口大小
        self.width = pixmap.width()  # 获取背景图的宽度1600
        self.height = pixmap.height()  # 获取背景图的高度900
        self.resize(self.width, self.height)  # 窗口自适应背景图的大小
        # 查看一下窗口尺寸
        print(self.width, self.height)
        # 调用方法，在方法体中创建所需控件
        self.setup_ui()

    def setup_ui(self):
        # 创建关闭系统按钮
        self.btn_close = QPushButton(self)
        self.btn_close.setStyleSheet("""
                    QPushButton{
                        border-image: url(./ui_img/return.png);  /* 按钮背景图片 */
                    }
                """)
        self.btn_close.resize(115, 29)
        self.btn_close.move(1136, 53)
        self.btn_close.clicked.connect(self.exit_click)

        # 创建实时视频显示标签
        self.label_video = QLabel(self)
        self.label_video.setStyleSheet("""
                            QLabel{
                                color:#FFFFFF;  /* 设置文字颜色 */
                                font-size:35px;  /* 设置字体大小 */
                                background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                            }
                        """)
        self.label_video.resize(424, 428)
        self.label_video.move(124, 210)

        # 创建人员数量实时显示标签
        self.label_person = QLabel(self)
        self.label_person.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_person.setStyleSheet("""
                            QLabel{
                                color:#FFFFFF;  /* 设置文字颜色 */
                                font-size:55px;  /* 设置字体大小 */
                                background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                            }
                        """)
        self.label_person.resize(56, 56)
        self.label_person.move(756, 204)

        # 创建吸烟显示标签
        self.label_smoking = QLabel(self)
        self.label_smoking.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_smoking.setStyleSheet("""
                                            QLabel{
                                                color:#FFFFFF;  /* 设置文字颜色 */
                                                font-size:55px;  /* 设置字体大小 */
                                                background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                            }
                                        """)
        self.label_smoking.resize(56, 56)
        self.label_smoking.move(1009, 204)

        # 创建自行车显示标签
        self.label_bicycle = QLabel(self)
        self.label_bicycle.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_bicycle.setStyleSheet("""
                                            QLabel{
                                                color:#FFFFFF;  /* 设置文字颜色 */
                                                font-size:55px;  /* 设置字体大小 */
                                                background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                            }
                                        """)
        self.label_bicycle.resize(56, 56)
        self.label_bicycle.move(756, 364)

        # 创建电动车显示标签
        self.label_motorcycle = QLabel(self)
        self.label_motorcycle.setAlignment(Qt.AlignCenter)  # 使文字垂直居中显示
        self.label_motorcycle.setStyleSheet("""
                                    QLabel{
                                        color:#FFFFFF;  /* 设置文字颜色 */
                                        font-size:55px;  /* 设置字体大小 */
                                        background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                    }
                                """)
        self.label_motorcycle.resize(56, 56)
        self.label_motorcycle.move(1009, 364)

        # 创建结果显示标签
        self.label_result = QLabel(self)
        self.label_result.setStyleSheet("""
                                            QLabel{
                                                color:#FFFFFF;  /* 设置文字颜色 */
                                                font-size:35px;  /* 设置字体大小 */
                                                background-color:rgba(255, 255, 255, 0);  /* 白色背景，透明度 */
                                            }
                                        """)
        self.label_result.resize(256, 96)
        self.label_result.move(792, 560)

        # 创建定时器
        self.timer = QTimer(self)
        # 连接定时器的 timeout 信号到 self.tick_callback 方法
        self.timer.timeout.connect(self.tick_callback)
        # 设置定时器间隔为 30 毫秒并启动定时器
        self.timer.start(30)

    def tick_callback(self):
        if self.n == 0:
            self.n += 1
            # 创建检测线程t2并运行，可以保证程序运行流畅
            t2 = threading.Thread(target=self.detect)
            # 创建线程守护
            t2.setDaemon(True)
            # 启动子线程
            t2.start()
        elif self.n == 1:
            self.n += 1
            # 创建界面更新线程t3并运行，可以保证程序运行流畅
            t3 = threading.Thread(target=self.update_gui)
            # 创建线程守护
            t3.setDaemon(True)
            # 启动子线程
            t3.start()

    def detect(self,
               weights=ROOT / 'weights/lift.pt',  # model path or triton URL
               source=ROOT / 'video.mp4',  # file/dir/URL/glob/screen/0(webcam)
               data=ROOT / 'data/coco_escalator.yaml',  # dataset.yaml path
               imgsz=(640, 640),  # inference size (height, width)
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               view_img=True,  # show results
               save_txt=False,  # save results to *.txt
               save_conf=False,  # save confidences in --save-txt labels
               save_crop=False,  # save cropped prediction boxes
               nosave=True,  # do not save images/videos
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               visualize=False,  # visualize features
               update=False,  # update all models
               project=ROOT / 'runs/detect',  # save results to project/name
               name='exp',  # save results to project/name
               exist_ok=False,  # existing project/name ok, do not increment
               line_thickness=3,  # bounding box thickness (pixels)
               hide_labels=False,  # hide labels
               hide_conf=False,  # hide confidences
               half=False,  # use FP16 half-precision inference
               dnn=False,  # use OpenCV DNN for ONNX inference
               vid_stride=1,  # video frame-rate stride
               ):
        # 变量path_result和result_img的作用：可以将识别结果作为索引显示相应图片
        path_result = 'ui_img/'
        result_img = ['wxts.png', 'nosmoking.jpg', 'noev.png']
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        #############################################################
                        self.p = names[int(c)]
                        #############################################################

                    # Write results
                    self.smoking = 0
                    self.motorcycle = 0
                    self.bicycle = 0
                    self.person = 0
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if int(cls) == 0:
                                self.smoking += 1
                            if int(cls) == 1:
                                self.motorcycle += 1
                            if int(cls) == 2:
                                self.bicycle += 1
                            if int(cls) == 3:
                                self.person += 1
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                else:
                    self.smoking = 0
                    self.motorcycle = 0
                    self.bicycle = 0
                    self.person = 0
                    self.p = None

                # Stream results
                im0 = annotator.result()
                if view_img:
                    self.n += 1
                    # im0表示每一帧检测的画面
                    self.im1 = im0
                    time.sleep(0.02)
                    if self.p == 'smoking':
                        outcome = 1
                    elif self.p == 'motorcycle':
                        outcome = 2
                    elif self.p == 'bicycle':
                        outcome = 0
                    elif self.p == 'person':
                        outcome = 0
                    else:
                        outcome = 0

                    # 根据返回结果outcome的值获取图片路径
                    p = path_result + result_img[outcome]
                    self.result = cv2.imread(p, cv2.IMREAD_COLOR)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    # 定义更新显示的方法,在子线程中调用
    def update_gui(self):
        while True:
            # 调用gui_util.py中的show_image_on_label函数在指定区域显示结果
            self.label_smoking.setText(str(self.smoking))  # 实时更新吸烟者数量
            self.label_motorcycle.setText(str(self.motorcycle))  # 实时更新电动车数量
            self.label_bicycle.setText(str(self.bicycle))  # 实时更新自行车数量
            self.label_person.setText(str(self.person))  # 实时更新人员数量
            show_image_on_label(self.im1, self.label_video, 424, 428)  # 实时摄像头
            show_image_on_label(self.result, self.label_result, 256, 96)  # 实时显示检测结果
            # 增加延时，降低卡顿。
            time.sleep(0.2)

    # 定义控制窗口关闭的方法
    def exit_click(self):
        time.sleep(1)
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
