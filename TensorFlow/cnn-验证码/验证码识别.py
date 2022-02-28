import numpy as np
import cv2 as cv2
import tensorflow as tf
import os
import random
tf.compat.v1.disable_eager_execution()
from tensorflow.python.keras import layers
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import load_img
tf.compat.v1.disable_eager_execution()
# config = tf.compat.v1.ConfigProto()

class GetMnist(object):
    def __init__(self):

        self.load_images_size=(30,100)
        self.max_captcha = 4
        self.char_set_len = 63
        self.image_dir= 'verify/'
        self.rate = 0.2

    def vec2text(self, vec):
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

    def text2vec(self, text):

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
    def load_data(self):
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
        imgs = os.listdir(self.image_dir)
        # 打乱图片顺序
        random.shuffle(imgs)
        # 数据集总共个数
        imgs_num = len(imgs)
        # 按照比例求出测试集个数
        test_num = int(imgs_num * self.rate / (1 + self.rate))
        # 测试集，测试数据的路径
        test_imgs = imgs[:test_num]
        # 根据文件名获取测试集标签
        test_labels = list(map(lambda x: x.split('.')[0], test_imgs))
        # 训练集
        train_imgs = imgs[test_num:]
        # 根据文件名获取训练集标签
        train_labels = list(map(lambda x: x.split('.')[0], train_imgs))

        x_test=np.array([img_to_array(load_img(os.path.join(self.image_dir,i),target_size=self.load_images_size)) for i in test_imgs])
        x_train=np.array([img_to_array(load_img(os.path.join(self.image_dir,i),target_size=self.load_images_size)) for i in train_imgs])
        y_test=np.array([ self.text2vec(i) for i in test_labels])
        y_train=np.array([ self.text2vec(i) for i in train_labels])
        return x_train/255.0,y_train, x_test/255.0,y_test






class CNNCode(object):
    def __init__(self,x_train, y_train,x_test, y_test):
        self.height = 30
        self.width = 100
        self.max_captcha = 4
        self.char_set_len = 63
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test


    def build_model(self):
        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=self.x_train[0].shape),

            tf.keras.layers.Conv2D(
                filters=64,
                input_shape=self.x_train[0].shape,
                kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
                                   ),
            # tf.keras.layers.Conv2D(
            #     filters=64, kernel_size=3, strides=1, padding="same",
            #     data_format="channels_last", activation=tf.nn.relu
            # ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            # 池化层1 2*2窗口  strides=2
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),

            # 卷积层2 64个 5*5*32的filter，strides=1  padding=same
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),
            # 卷积层2 128个 5*5*32的filter，strides=1  padding=same
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            # 池化层2 2*2窗口  strides=2 [none 8 8 64]
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),

            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            # 池化层2 2*2窗口  strides=2 [none 8 8 64]
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            # 池化层2 2*2窗口  strides=2 [none 8 8 64]
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),
            # tf.keras.layers.BatchNormalization(),

            # 全连接层神经网络
            # [none 8 8 64] ----> [None,8*8*64]
            tf.keras.layers.Flatten(),
            # 1024个神经元网络
            tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            # tf.keras.layers.Dense(1024, activation=tf.nn.relu),
            # 100个神经元网络

            tf.keras.layers.Dense(4 * 63, activation=tf.nn.softmax),

        ])
        return self.model
    def model_compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                          loss=tf.keras.losses.categorical_crossentropy,
                          metrics=['accuracy'])

    def fit_model(self):
        # filepath="/ckpt/test.h5"
        modelckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath='./ckpt/transfer_{epoch:02d}-{accuracy:.4f}.h5',
            # filepath=filepath,
            monitor='accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1

        )


        self.model.fit(x=self.x_train, y=self.y_train,
                  batch_size=32,
                  epochs=500,
                       callbacks=[modelckpt]
                       )


if __name__ == '__main__':
    data=GetMnist()
    # x_train, y_train,x_test, y_test=data.load_data()
    x_train, y_train,x_test, y_test=data.load_data()


    cnn=CNNCode(x_train, y_train,x_test, y_test)
    model=cnn.build_model()
    print(model.summary())
    cnn.model_compile()
    cnn.fit_model()

