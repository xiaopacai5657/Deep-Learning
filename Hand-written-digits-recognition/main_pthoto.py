import cv2 as cv
import numpy as np
import os
from Pre_treatment import get_number as g_n
import predict as pt
from time import time
from Pre_treatment import softmax

net = pt.get_net()  # 加载预训练的神经网络模型
# orig_path = r"./real_img_resize"
# img_list = os.listdir(orig_path)
# print(img_list)
#
# for img_name in img_list:
#     since = time()
#     img_path = os.path.join(orig_path, img_name)
#     img = cv.imread(img_path)
#     img_bw = g_n(img)
#     img_bw_c = img_bw.sum(axis=1) / 255
#     img_bw_r = img_bw.sum(axis=0) / 255
#     r_ind, c_ind = [], []
#     for k, r in enumerate(img_bw_r):
#         if r >= 5:
#             r_ind.append(k)
#     for k, c in enumerate(img_bw_c):
#         if c >= 5:
#             c_ind.append(k)
#     img_bw_sg = img_bw[c_ind[0]:c_ind[-1], r_ind[0]:r_ind[-1]]
#     leng_c = len(c_ind)
#     leng_r = len(r_ind)
#     side_len = leng_c + 20
#     add_r = int((side_len - leng_r) / 2)
#     img_bw_sg_bord = cv.copyMakeBorder(img_bw_sg, 10, 10, add_r, add_r, cv.BORDER_CONSTANT, value=[0, 0, 0])
#     # 展示图片
#     cv.imshow("img", img_bw)
#     cv.imshow("img_sg", img_bw_sg_bord)
#     c = cv.waitKey(1) & 0xff
#
#     img_in = cv.resize(img_bw_sg_bord, (28, 28))
#     result_org = pt.predict(img_in, net)
#     result = softmax(result_org)
#     best_result = result.argmax(dim=1).item()
#     best_result_num = max(max(result)).cpu().detach().numpy()
#     if best_result_num <= 0.5:
#         best_result = None
#
#     # 显示结果
#     img_show = cv.resize(img, (600, 600))
#     end_predict = time()
#     fps = np.ceil(1 / (end_predict - since))
#     font = cv.FONT_HERSHEY_SIMPLEX
#     cv.putText(img_show, "The number is:" + str(best_result), (1, 30), font, 1, (0, 0, 255), 2)
#     cv.putText(img_show, "Probability is:" + str(best_result_num), (1, 60), font, 1, (0, 255, 0), 2)
#     cv.putText(img_show, "FPS:" + str(fps), (1, 90), font, 1, (255, 0, 0), 2)
#     cv.imshow("result", img_show)
#     cv.waitKey(1)
#     print(result)
#     print("*" * 50)
#     print("The number is:", best_result)


img_path = r'./real_img/7.jpg'  # 定义输入图像的路径
since = time()  # 记录开始时间，用于计算FPS
img = cv.imread(img_path)  # 读取输入图像
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
img_bw_sg_bord = cv.copyMakeBorder(
    img_bw_sg, 10, 10, add_r, add_r,  # 上下各10像素，左右各add_r像素
    cv.BORDER_CONSTANT, value=[0, 0, 0]  # 黑色填充
)

# 调整图像尺寸为28x28（模型输入尺寸）
img_in = cv.resize(img_bw_sg_bord, (28, 28))

# 使用模型预测并处理结果
result_org = pt.predict(img_in, net)  # 获取原始预测结果（logits）
result = softmax(result_org)  # 应用softmax获取概率分布
best_result = result.argmax(dim=1).item()  # 找到概率最大的类别
best_result_num = max(max(result)).cpu().detach().numpy()  # 提取最大概率值

# 若概率≤0.5则认为结果不可靠
if best_result_num <= 0.5:
    best_result = None

# 显示结果
img_show = cv.resize(img, (600, 600))  # 缩放原图以便显示
end_predict = time()  # 记录结束时间
fps = np.ceil(1 / (end_predict - since))  # 计算FPS

# 添加文字到显示图像
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img_show, "The number is:" + str(best_result), (1, 30), font, 1, (0, 0, 255), 2)
cv.putText(img_show, "Probability is:" + str(best_result_num), (1, 60), font, 1, (0, 255, 0), 2)
cv.putText(img_show, "FPS:" + str(fps), (1, 90), font, 1, (255, 0, 0), 2)

# 打印结果
print(result)
print("*" * 50)
print("The number is:", best_result)

# 显示图像窗口
cv.imshow("img", img_bw)  # 预处理后的二值图
cv.imshow("img_sg", img_bw_sg_bord)  # 裁剪并添加边框后的图像
cv.imshow("result", img_show)  # 最终结果图像

# 等待用户按键关闭窗口
if cv.waitKey(0) & 0xFF == ord(' '):
    cv.destroyAllWindows()
