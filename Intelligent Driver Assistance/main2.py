from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import *
from qfluentwidgets import InfoBar, InfoBarPosition
from PyQt5.QtCore import Qt
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox
from PyQt5 import uic
import os
import platform
import sys
from pathlib import Path
import numpy as np
import time
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode

# 全局变量用于存储选择的视频路径
video_path = ""
# 全局变量用于存储对象
cap = cv2.VideoCapture()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 直接加载.ui文件
        uic.loadUi('GUI.ui', self)
        # 设置全屏显示
        # self.showFullScreen()
        # 设置窗口标题
        self.setWindowTitle("智能驾驶辅助系统")
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
        self.btn_exit.clicked.connect(self.exit_click)
        # 选择文件
        self.btn_open.clicked.connect(self.open)
        # 开始检测
        self.btn_ctr.clicked.connect(self.detect)
        # 设置文本控件的样式表
        self.TextEdit.setStyleSheet("""
                    QTextEdit{
                        font-size:18px;  /* 设置字体大小 */
                    }
                """)
        # 初始化检测状态
        self.detecting = False

    def open(self):
        # 创建文件对话框
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mov *.mkv)")
        # 打开文件对话框
        if file_dialog.exec_():
            # 获取选择的文件路径
            global video_path, cap  # 声明全局变量
            video_path = file_dialog.selectedFiles()[0]
            print(video_path)
            # 显示成功消息
            self.createSuccessInfoBar("成功", f"已选择视频：{os.path.basename(video_path)}")
            # 确保视频路径有效
            if not os.path.exists(video_path):
                self.createErrorInfoBar("错误", "视频路径不存在，请重新选择!")
                return
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            # 读取第一帧
            ret, frame = cap.read()
            # 释放视频资源
            cap.release()
            # 检查是否成功读取帧
            if not ret:
                self.createErrorInfoBar("错误", "无法读取视频帧，请检查视频文件!")
                return
            # 显示第一帧图像
            self.show_image_on_label(frame, self.label_result, 951, 571)

    # 获取车道线检测的选中状态
    def detect(self):
        # 检查是否已选择视频文件
        global video_path
        if 'video_path' not in globals() or not os.path.exists(video_path):
            self.createErrorInfoBar("错误", "请先选择视频文件")
            return

        # 获取车道线检测的选中状态
        lane_status = self.btn_lane.isChecked()
        # 获取路况检测的选中状态
        road_status = self.btn_road.isChecked()
        # 检查是否未选择任何模式
        if not lane_status and not road_status:
            self.createErrorInfoBar("错误", "未选择检测模式")
            return

        # 切换检测状态
        self.detecting = not self.detecting
        if self.detecting:
            self.btn_ctr.setText("结束检测")
            # 根据选择执行对应的检测
            if lane_status:
                self.createSuccessInfoBar("成功", "车道线检测已开始。")
                # 调用车道线检测函数
                # self.start_lane_detection()
                t2 = threading.Thread(target=self.start_lane_detection)
                # 创建线程守护
                t2.setDaemon(True)
                # 启动子线程
                t2.start()
            elif road_status:
                self.createSuccessInfoBar("成功", "路况检测已开始。")
                # 调用路况检测函数
                self.start_road_detection()
                # t3 = threading.Thread(target=self.start_road_detection)
                # # 创建线程守护
                # t3.setDaemon(True)
                # # 启动子线程
                # t3.start()
        else:
            self.btn_ctr.setText("开始检测")
            # 停止检测
            self.stop_detection()

    # 车道线检测函数
    def start_lane_detection(self):
        if self.detecting == False:
            return
            # 检查是否已选择视频文件
        global video_path, cap
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.createErrorInfoBar("错误", "没有正确打开视频文件!")
            return

        while cap.isOpened():
            try:
                ret, img = cap.read()
                if not ret:
                    break

                start = time.time()
                # 图像预处理
                grap = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                blur_grap = cv2.GaussianBlur(grap, (3, 3), 0)
                canny_image = self.canny_func(blur_grap)

                # 设置ROI区域
                left_bottom = [0, canny_image.shape[0]]
                right_bottom = [canny_image.shape[1], canny_image.shape[0]]
                left_top = [canny_image.shape[1] / 3, canny_image.shape[0] / 1.5]
                right_top = [canny_image.shape[1] / 3 * 2, canny_image.shape[0] / 1.5]
                vertices = np.array([left_top, right_top, right_bottom, left_bottom], np.int32)
                roi_image = self.roi_mask(canny_image, vertices)

                # 霍夫变换检测直线
                line_img = self.hough_func(roi_image)

                # 绘制车道线
                img = self.draw_lines(img, line_img)

                end = time.time()
                detect_fps = round(1.0 / (end - start + 0.00001), 2)

                # 添加文本信息
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.putText(img, f'Lane line detection | FPS: {detect_fps}',
                                  (40, 40), font, 0.7, (0, 0, 0), 2)
                # 显示结果
                self.show_image_on_label(img, self.label_result, 951, 571)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            except Exception as e:
                self.createErrorInfoBar("错误", f"{str(e)}")
                break

    # 路况检测函数
    def start_road_detection(self,
                             weights='yolov5s-seg.pt',  # model.pt path(s)
                             data='data/coco128.yaml',  # dataset.yaml path
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
                             project='runs/predict-seg',  # save results to project/name
                             name='exp',  # save results to project/name
                             exist_ok=True,  # existing project/name ok, do not increment
                             line_thickness=3,  # bounding box thickness (pixels)
                             hide_labels=False,  # hide labels
                             hide_conf=False,  # hide confidences
                             half=False,  # use FP16 half-precision inference
                             dnn=False,  # use OpenCV DNN for ONNX inference
                             vid_stride=1,  # video frame-rate stride
                             retina_masks=False,
                             ):
        # 检查是否已选择视频文件
        global video_path
        if video_path == "":
            self.createErrorInfoBar("错误", "请先选择视频文件")
            return
        source = str(video_path)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)  # 此处有修改
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            if self.detecting == False:
                video_path = ""
                break
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, proto = model(im, augment=augment, visualize=visualize)[:2]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

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
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Segments
                    if save_txt:
                        segments = reversed(masks2segments(masks))
                        segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        #############################################################
                        self.p = names[int(c)]
                        #############################################################

                    # Mask plotting
                    annotator.masks(masks,
                                    colors=[colors(x, True) for x in det[:, 5]],
                                    im_gpu=None if retina_masks else im[i])

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        if save_txt:  # Write to file
                            segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                            line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                else:
                    self.p = None

                # Stream results
                im0 = annotator.result()
                # 车道线检测
                start = time.time()
                # 图像预处理
                grap = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
                blur_grap = cv2.GaussianBlur(grap, (3, 3), 0)
                canny_image = self.canny_func(blur_grap)

                # 设置ROI区域
                left_bottom = [0, canny_image.shape[0]]
                right_bottom = [canny_image.shape[1], canny_image.shape[0]]
                left_top = [canny_image.shape[1] / 3, canny_image.shape[0] / 1.5]
                right_top = [canny_image.shape[1] / 3 * 2, canny_image.shape[0] / 1.5]
                vertices = np.array([left_top, right_top, right_bottom, left_bottom], np.int32)
                roi_image = self.roi_mask(canny_image, vertices)

                # 霍夫变换检测直线
                line_img = self.hough_func(roi_image)

                # 绘制车道线
                im0 = self.draw_lines(im0, line_img)

                end = time.time()
                detect_fps = round(1.0 / (end - start + 0.00001), 2)

                # 添加文本信息
                font = cv2.FONT_HERSHEY_SIMPLEX
                im0 = cv2.putText(im0, f'Traffic condition monitoring | FPS: {detect_fps}',
                                  (40, 40), font, 0.7, (0, 0, 0), 2)
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        # cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.resizeWindow(str(p), 1, 1)
                    if self.p == "person":
                        self.show_info_on_text(
                            "检测到行人！请注意提前减速慢行，保持安全车距，随时准备停车避让，避免发生碰撞事故。")
                    elif self.p == "bicycle":
                        self.show_info_on_text(
                            "检测到自行车！请提前减速，保持足够的横向距离，避免因距离过近引发剐蹭事故。")
                    elif self.p == "car":
                        self.show_info_on_text(
                            "前方检测到车辆！请保持安全车距，注意观察前车动态，避免急刹车和频繁变道，防止追尾事故发生。")
                    elif self.p == "motorcycle":
                        self.show_info_on_text(
                            "注意摩托车！摩托车速度较快且车身较窄，容易在视线盲区出现，务必提高警惕，保持安全距离，切勿随意变道挤占其行驶空间。")
                    elif self.p == "bus":
                        self.show_info_on_text(
                            "检测到公交车！公交车体型庞大，停靠站点频繁，请注意观察其转向灯和刹车灯，保持足够车距，避免因视线遮挡而发生危险。")
                    elif self.p == "truck":
                        self.show_info_on_text(
                            "检测到大型车辆！卡车车身长、盲区范围广，请勿长时间在其盲区行驶，超车时要果断迅速，保持安全距离，谨防其突然转向或刹车。")
                    elif self.p == "traffic light":
                        self.show_info_on_text(
                            "注意交通信号灯！提前观察信号灯变化，减速慢行，严格遵守交通规则，避免因抢行闯红灯引发交通事故。")
                    elif self.p == "stop sign":
                        self.show_info_on_text(
                            "前方有停车标志！请立即减速停车，仔细观察四周交通情况，确认安全后再继续行驶，切勿忽视停车标志强行通过。")
                    else:
                        self.show_info_on_text("")
                    self.show_image_on_label(im0, self.label_result, 951, 571)

                    if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                        exit()

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

    # 结束检测
    def stop_detection(self):
        global cap, video_path
        self.detecting = False
        # 释放视频捕获对象资源
        if cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            video_path = ""
            self.createSuccessInfoBar("成功", "已结束检测。")
        time.sleep(0.5)
        # 清空提示信息
        self.TextEdit.clear()
        # 清空显示的图像
        width, height = 951, 571
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(255, 255, 255, 0.75))
        self.label_result.setPixmap(pixmap)

    # 定义控制窗口关闭的方法
    def exit_click(self):
        time.sleep(1)
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    def canny_func(self, blur_gray, canny_lthreshold=150, canny_hthreshold=250):
        """
        使用Canny算子进行边缘检测
        :param blur_gray: 灰度化且高斯平滑后的图像
        :param canny_lthreshold: Canny算子低阈值
        :param canny_hthreshold: Canny算子高阈值
        :return: 边缘检测后的图像
        """
        edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
        return edges

    def roi_mask(self, img, vertices):
        """
        设置ROI区域
        :param img: 输入图像
        :param vertices: ROI区域顶点坐标
        :return: 应用ROI掩码后的图像
        """
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            mask_color = (255,) * channel_count
        else:
            mask_color = 255
        cv2.fillPoly(mask, [vertices], mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def hough_func(self, roi_image, rho=1, theta=np.pi / 180, threshold=15, min_line_lenght=40, max_line_gap=20):
        """
        使用霍夫变换检测直线
        :param roi_image: ROI区域图像
        :param rho: 距离分辨率（以像素为单位）
        :param theta: 角度分辨率（以弧度为单位）
        :param threshold: 累加器阈值
        :param min_line_lenght: 最小直线长度
        :param max_line_gap: 最大直线间隙
        :return: 检测到的直线列表
        """
        line_img = cv2.HoughLinesP(roi_image, rho, theta, threshold, minLineLength=min_line_lenght,
                                   maxLineGap=max_line_gap)
        return line_img

    def draw_lines(self, img, lines, color=[0, 0, 255], thickness=2):
        """
        绘制车道线并填充中间区域
        :param img: 原始图像
        :param lines: 检测到的直线列表
        :param color: 线条颜色
        :param thickness: 线条粗细
        :return: 绘制车道线后的图像
        """
        left_lines_x = []
        left_lines_y = []
        right_lines_x = []
        right_lines_y = []
        line_y_max = 0
        line_y_min = 999

        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    line_y_max = max(line_y_max, y1, y2)
                    line_y_min = min(line_y_min, y1, y2)
                    k = (y2 - y1) / (x2 - x1)
                    if k < -0.3:
                        left_lines_x.extend([x1, x2])
                        left_lines_y.extend([y1, y2])
                    elif k > 0.3:
                        right_lines_x.extend([x1, x2])
                        right_lines_y.extend([y1, y2])

            # 最小二乘直线拟合
            if left_lines_x and left_lines_y:
                left_line_k, left_line_b = np.polyfit(left_lines_x, left_lines_y, 1)
            else:
                left_line_k, left_line_b = 0, 0
            if right_lines_x and right_lines_y:
                right_line_k, right_line_b = np.polyfit(right_lines_x, right_lines_y, 1)
            else:
                right_line_k, right_line_b = 0, 0

            # 根据直线方程和最大、最小的y值反算对应的x
            if left_line_k != 0:
                cv2.line(img,
                         (int((line_y_max - left_line_b) / left_line_k), line_y_max),
                         (int((line_y_min - left_line_b) / left_line_k), line_y_min),
                         color, thickness)
            if right_line_k != 0:
                cv2.line(img,
                         (int((line_y_max - right_line_b) / right_line_k), line_y_max),
                         (int((line_y_min - right_line_b) / right_line_k), line_y_min),
                         color, thickness)

            # 填充车道线中间区域
            zero_img = np.zeros((img.shape), dtype=np.uint8)
            if left_line_k != 0 and right_line_k != 0:
                polygon = np.array([
                    [int((line_y_max - left_line_b) / left_line_k), line_y_max],
                    [int((line_y_max - right_line_b) / right_line_k), line_y_max],
                    [int((line_y_min - right_line_b) / right_line_k), line_y_min],
                    [int((line_y_min - left_line_b) / left_line_k), line_y_min]
                ])
                cv2.fillConvexPoly(zero_img, polygon, color=(0, 255, 0))

            alpha = 1
            beta = 0.2
            gamma = 0
            img = cv2.addWeighted(img, alpha, zero_img, beta, gamma)

        except Exception as e:
            print(str(e))
        return img

    # 定义在标签控件上显示识别结果
    # 参数：img - 输入图像，label - 目标标签控件，w - 显示宽度，h - 显示高度
    def show_image_on_label(self, img, label, w, h):
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

    # 定义在信息框中显示提示信息
    def show_info_on_text(self, info):
        self.TextEdit.clear()
        self.TextEdit.append(str(info))

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
