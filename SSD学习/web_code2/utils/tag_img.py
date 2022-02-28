import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

classes_name = ['clothes', 'pants', 'shoes', 'watch', 'phone',
                             'audio', 'computer', 'books']


def tag_picture(img, outputs):
    """
    对图片预测物体进行画图显示
    :param images_data: N个测试图片数据
    :param outputs: 每一个图片的预测结果
    :return:
    """
    # 1、先获取每张图片6列中的结果

    # 通过i获取图片label, location, xmin, ymin, xmax, ymax
    pre_label = outputs[0][:, 0]
    pre_conf = outputs[0][:, 1]
    pre_xmin = outputs[0][:, 2]
    pre_ymin = outputs[0][:, 3]
    pre_xmax = outputs[0][:, 4]
    pre_ymax = outputs[0][:, 5]

    top_indices = [i for i, conf in enumerate(pre_conf) if conf >= 0.3]
    top_conf = pre_conf[top_indices]
    top_label_indices = pre_label[top_indices].tolist()
    top_xmin = pre_xmin[top_indices]
    top_ymin = pre_ymin[top_indices]
    top_xmax = pre_xmax[top_indices]
    top_ymax = pre_ymax[top_indices]

    # print("pre_label:{}, pre_loc:{}, pre_xmin:{}, pre_ymin:{},pre_xmax:{},pre_ymax:{}".
    #       format(tag_label, tag_loc, tag_xmin, tag_ymin, tag_xmax, tag_ymax))

    # 对于每张图片的结果进行标记
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))

        # 获取该图片预测概率，名称，定义显示颜色
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = classes_name[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        # 显示方框
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        # 左上角显示概率以及名称
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

        # plt.show()
    image_io = BytesIO()
    plt.savefig(image_io, format='png')
    image_io.seek(0)
    return image_io