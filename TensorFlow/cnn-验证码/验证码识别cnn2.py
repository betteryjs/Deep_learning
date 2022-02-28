import numpy as np
import cv2 as cv2
import tensorflow as tf
import os
import random
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import load_img
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
#
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
# # config.gpu_options.per_process_gpu_memory_fraction = 0.6  #限制GPU内存占用率
#
session = tf.compat.v1.InteractiveSession(config=config)





class CNNCode(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.max_captcha = 4
        self.char_set_len = 63
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.base_model = VGG16(weights='imagenet', include_top=False)

        self.train_generator = ImageDataGenerator(
            rescale=1.0/255.0,
            zca_whitening=True,
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)



    def vec2text(self,vec):
        """
        向量转文本
        Parameters:
            vec:向量
        Returns:
            文本
        """
        char_pos = vec.nonzero()[0]
        text = []
        for c in char_pos:
            char_idx = c % self.char_set_len
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    def text2vec(self,text):

        """
        文本转向量
        Parameters:
            text:文本
        Returns:
            vector:向量
        """
        if len(text) > 4:
            raise ValueError('验证码最长4个字符')

        vector = np.zeros(4 * self.char_set_len)

        def char2pos(c):
            if c == '_':
                k = 62
                return k
            k = ord(c) - 48
            if k > 9:
                k = ord(c) - 55
                if k > 35:
                    k = ord(c) - 61
                    if k > 61:
                        raise ValueError('No Map')
            return k

        for i, c in enumerate(text):
            idx = i * self.char_set_len + char2pos(c)
            vector[idx] = 1
        return vector

    def get_imgs(self,rate=0.2):
        """
        获取图片，并划分训练集和测试集
        Parameters:
            rate:测试集 2 和训练集 10 的比例，即测试集个数/训练集个数
        Returns:
            test_imgs:测试集
            test_labels:测试集标签
            train_imgs:训练集
            train_labels:训练集标签
        """
        # 读取图片
        imgs = os.listdir('verify/')
        # 打乱图片顺序
        random.shuffle(imgs)
        # 数据集总共个数
        imgs_num = len(imgs)
        # 按照比例求出测试集个数
        test_num = int(imgs_num * rate / (1 + rate))
        # 测试集，测试数据的路径
        test_imgs = imgs[:test_num]
        # 根据文件名获取测试集标签
        test_labels = list(map(lambda x: x.split('.')[0], test_imgs))
        # 训练集
        train_imgs = imgs[test_num:]
        # 根据文件名获取训练集标签
        train_labels = list(map(lambda x: x.split('.')[0], train_imgs))

        self.train_got(test_imgs, test_labels, train_imgs, train_labels)

    def train_got(self,test_imgs, test_labels, train_imgs, train_labels):

        self.x_test = np.zeros([np.array(test_imgs).shape[0], 224,224,3])
        self.y_test = np.zeros([np.array(test_labels).shape[0], self.max_captcha * self.char_set_len])
        for index, train in enumerate(test_imgs):
            # 黑白图片
            img = load_img('./verify/' + train, target_size=(224, 224))
            img =  img_to_array(img)
            # img = np.mean(cv2.imread('./verify/' + train), axis=-1)
            # 将多维降维1维
            self.x_test[index] = img
        for index, label in enumerate(test_labels):
            self.y_test[index, :] = self.text2vec(label)

        self.x_train = np.zeros([np.array(train_imgs).shape[0],224,224,3])
        self.y_train = np.zeros([np.array(train_labels).shape[0], self.max_captcha * self.char_set_len])
        for index, train in enumerate(train_imgs):
            # 黑白图片
            img = load_img('./verify/' + train, target_size=(224, 224))
            img =  img_to_array(img)
            # img = np.mean(cv2.imread('./verify/' + train), axis=-1)
            # 将多维降维1维
            self.x_train[index] = img
        for index, label in enumerate(train_labels):
            self.y_train[index, :] = self.text2vec(label)

        # self.x_test = x_test.reshape([-1,self.height, self.width, 3])
        # self.x_train = x_train.reshape([-1, self.height, self.width, 3])


    def fit_model(self,model):

        epochs=100
        for e in range(epochs):
            print('Epoch', e)
            batches = 0
            for x_batch, y_batch in self.train_generator.flow(self.x_train, self.y_train, batch_size=16):
                model.fit(x_batch, y_batch)

    def refine_base_model(self):
        """

        微调VGG结构  5个blocks后加入+全局平均池化(减少迁移学习参数数量)+两个全连接层
        :return:
        """

        # 1.获取原notop模型的输出
        # [?,?,?,512]
        x = self.base_model.outputs[0]
        # 2.在输出后面增加结构

        # [?,?,?,512] ----> [?,512]
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # 3. 定义新的模型

        x = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(x)
        y_perdict = tf.keras.layers.Dense(4*63, activation=tf.nn.softmax)(x)

        # model
        transfer_model = tf.keras.models.Model(inputs=self.base_model.inputs, outputs=y_perdict)

        return transfer_model
    def freeze_model(self):
        """

        冻结VGG初始模型 (5blocks) 不进行训练
        :return:
        """
        # self.base_model.layers 获取所有层的列表
        for layer in self.base_model.layers:
            layer.trainable = False
    def model_compile(self, model):
        """

        编译模型
        :return:
        """

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )
        return None

if __name__ == '__main__':
    cnn=CNNCode()
    cnn.get_imgs()
    model=cnn.refine_base_model()
    cnn.freeze_model()
    cnn.model_compile(model)
    cnn.fit_model(model)
