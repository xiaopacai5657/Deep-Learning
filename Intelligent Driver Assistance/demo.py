import os
import platform
import sys
from pathlib import Path
import cv2
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


@smart_inference_mode()
def main(
        weights='yolov5s-seg.pt',  # model.pt path(s)
        source='example/video_3.mp4',  # file/dir/URL/glob/screen/0(webcam)
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
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
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

            # Stream results
            im0 = annotator.result()
            # 车道线检测
            start = time.time()
            # 图像预处理
            grap = cv2.cvtColor(im0, cv2.COLOR_RGB2GRAY)
            blur_grap = cv2.GaussianBlur(grap, (3, 3), 0)
            canny_image = canny_func(blur_grap)

            # 设置ROI区域
            left_bottom = [0, canny_image.shape[0]]
            right_bottom = [canny_image.shape[1], canny_image.shape[0]]
            left_top = [canny_image.shape[1] / 3, canny_image.shape[0] / 1.5]
            right_top = [canny_image.shape[1] / 3 * 2, canny_image.shape[0] / 1.5]
            vertices = np.array([left_top, right_top, right_bottom, left_bottom], np.int32)
            roi_image = roi_mask(canny_image, vertices)

            # 霍夫变换检测直线
            line_img = hough_func(roi_image)

            # 绘制车道线
            im0 = draw_lines(im0, line_img)

            end = time.time()
            detect_fps = round(1.0 / (end - start + 0.00001), 2)

            # 添加文本信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            im0 = cv2.putText(im0, f'Lane detect v1.0 | Gao Wenpeng | FPS: {detect_fps}',
                              (40, 40), font, 0.7, (0, 0, 0), 2)
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
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


def canny_func(blur_gray, canny_lthreshold=150, canny_hthreshold=250):
    """
    使用Canny算子进行边缘检测
    :param blur_gray: 灰度化且高斯平滑后的图像
    :param canny_lthreshold: Canny算子低阈值
    :param canny_hthreshold: Canny算子高阈值
    :return: 边缘检测后的图像
    """
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    return edges


def roi_mask(img, vertices):
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


def hough_func(roi_image, rho=1, theta=np.pi / 180, threshold=15, min_line_lenght=40, max_line_gap=20):
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
    line_img = cv2.HoughLinesP(roi_image, rho, theta, threshold, minLineLength=min_line_lenght, maxLineGap=max_line_gap)
    return line_img


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
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
        print("NO detection")

    return img


if __name__ == "__main__":
    main()
