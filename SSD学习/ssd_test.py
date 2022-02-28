from nets.ssd_net import SSD300
from utils.ssd_utils import BBoxUtility
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import load_img
import os
from imageio import imread
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
模型预测流程
SSD300模型输入以及加载参数
读取多个本地路径测试图片，preprocess_input以及保存图像像素值（显示需要）
模型预测结果，得到7308个prior box
进行非最大抑制算法处理
"""


class SSDTest(object):
    def __init__(self):
        # 定义识别类别
        self.classes_name = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                             'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                             'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                             'Sheep', 'Sofa', 'Train', 'Tvmonitor']

        # 定义算法模型的输入参数 1 是背景
        self.classes_nums = len(self.classes_name) + 1
        self.input_shape = (300, 300, 3)

    def test(self):
        """

        对于输入图片进行预测
        :return:
        """

        # SSD300模型输入以及加载参数
        model = SSD300(self.input_shape, num_classes=self.classes_nums)
        model.load_weights('./ckpt/pre_trained/weights_SSD300.hdf5', by_name=True)

        # 读取多个本地路径测试图片，preprocess_input以及保存图像像素值（显示需要）
        feature = []
        images_data = []
        for path in os.listdir('images'):
            image_path = os.path.join('images', path)
            # 1. 输入SSD网络中 数组
            print(image_path)
            image = load_img(image_path, target_size=(self.input_shape[0], self.input_shape[1]))
            image = img_to_array(image)
            feature.append(image)
            # 2. 读取图片二进制数据 matplotlib 显示使用

            images_data.append(imread(image_path))
        # 模型预测结果，得到7308个 prior box

        inputs = preprocess_input(np.asarray(feature))
        predict = model.predict(inputs)
        # res.shape (2,7308,3) 2代表图片数量 7308代表每个图片预测的default boxes 数量
        # 33: 4(位置)+21(预测概率)+8(其他相关配值参数)
        print(predict.shape)

        # 进行非最大抑制算法处理 NMS 21个类别

        bbox = BBoxUtility(self.classes_nums)
        resule = bbox.detection_out(predict)
        # (200,6) 200:200个候选框 每个候选框的位置类别
        print(resule[0].shape, resule[1].shape)
        return resule, images_data

    def tag_picture(self, images_data, outputs):
        """

        显示预测结构到图片中
        :return:
        """
        # 1. 获取每张图片的预测结构中的值
        for index, image in enumerate(images_data):
            # 获取res当中对应的结构 label, confidence : 预测的概率值, xmin, ymin, xmax, ymax
            pre_label = outputs[index][:, 0]
            pre_confidence = outputs[index][:, 1]
            pre_xmin = outputs[index][:, 2]
            pre_ymin = outputs[index][:, 3]
            pre_xmax = outputs[index][:, 4]
            pre_ymax = outputs[index][:, 5]
            print("label:{}, probability:{}, xmin:{}, ymin:{}, xmax:{}, ymax:{}".
                  format(pre_label, pre_confidence, pre_xmin, pre_ymin, pre_xmax, pre_ymax))
            # 由于检测出的物体还是很多的 所以进行显示过滤
            top_indices=[i for i,conf in enumerate(pre_confidence) if conf>0.6]

            top_conf = pre_confidence[top_indices]
            top_label_indices = pre_label[top_indices].tolist()
            top_xmin = pre_xmin[top_indices]
            top_ymin = pre_ymin[top_indices]
            top_xmax = pre_xmax[top_indices]
            top_ymax = pre_ymax[top_indices]
            print("top_conf:{}, top_label_indices:{}, pre_xmin:{}, pre_ymin:{},pre_xmax:{},pre_ymax:{}".
                  format(top_conf, top_label_indices, top_xmin, top_ymin, top_xmax, top_ymax))


            colors=sns.color_palette('hls',21)
            plt.imshow(image / 255.0)
            currentAxis = plt.gca()
            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(round(top_ymax[i] * image.shape[0]))

                # 获取该图片预测概率，名称，定义显示颜色
                score = top_conf[i]
                label = int(top_label_indices[i])
                label_name = self.classes_name[label - 1]
                display_txt = '{:0.2f}, {}'.format(score, label_name)
                coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                color = colors[label]
                # 显示方框
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # 左上角显示概率以及名称
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

            plt.show()

        return None


if __name__ == '__main__':
    ssd = SSDTest()
    outputs, images_data = ssd.test()
    # 显示图片
    ssd.tag_picture(images_data, outputs)
