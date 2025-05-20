import cv2
import numpy as np
import time


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


# 开始检测
video_file = "./example/video_3.mp4"
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("没有正确打开视频文件")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 保存视频
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("./output.mp4", fourcc, fps, (width, height), 1)

while cap.isOpened():
    try:
        ret, img = cap.read()
        if not ret:
            break

        start = time.time()
        # 图像预处理
        grap = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        img = draw_lines(img, line_img)

        end = time.time()
        detect_fps = round(1.0 / (end - start + 0.00001), 2)

        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, f'Lane line detection | FPS: {detect_fps}',
                          (40, 40), font, 0.7, (0, 0, 0), 2)

        # 保存并显示结果
        writer.write(img)
        cv2.imshow('lane_detect', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(str(e))
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
