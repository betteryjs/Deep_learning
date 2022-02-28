import numpy as np

import cv2 as cv2

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from tensorflow import keras
import os
import random
tf.compat.v1.disable_eager_execution()


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# 预先写几个方法
# 方法一就是获取文件夹下的数据
def get_imgs(rate=0.2):
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

    return test_imgs, test_labels, train_imgs, train_labels



def text2vec(text):
    char_set_len = 63
    """
    文本转向量
    Parameters:
        text:文本
    Returns:
        vector:向量
    """
    if len(text) > 4:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(4 * char_set_len)
    def char2pos(c):
        if c =='_':
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
        idx = i * char_set_len + char2pos(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    char_set_len = 63
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
        char_idx = c % char_set_len
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




test_imgs, test_labels, train_imgs, train_labels=get_imgs()

def train_got(test_imgs, test_labels, train_imgs, train_labels):

    height = 30
    width = 100
    max_captcha = 4
    char_set_len = 63
    x_test = np.zeros([np.array(test_imgs).shape[0], height*width])
    y_test = np.zeros([np.array(test_labels).shape[0], max_captcha*char_set_len])
    for index, train in enumerate(test_imgs):
                # 黑白图片
                img = np.mean(cv2.imread('./verify/' + train), axis = -1)
                # 将多维降维1维
                x_test[index,:] = img.flatten() / 255.0
    for index, label in enumerate(test_labels):
                y_test[index,:] = text2vec(label)

    x_train = np.zeros([np.array(train_imgs).shape[0], height*width])
    y_train = np.zeros([np.array(train_labels).shape[0], max_captcha*char_set_len])
    for index, train in enumerate(train_imgs):
                # 黑白图片
                img = np.mean(cv2.imread('./verify/' + train), axis = -1)
                # 将多维降维1维
                x_train[index,:] = img.flatten() / 255.0
    for index, label in enumerate(train_labels):
                y_train[index,:] = text2vec(label)

    x_test=x_test.reshape(-1,30,100,1)
    x_train=x_train.reshape(-1,30,100,1)
    return x_test ,x_train  ,test_labels,train_labels

x_test ,x_train,y_test,y_train=train_got(test_imgs, test_labels, train_imgs, train_labels)




model = tf.keras.models.Sequential([
            # 卷积层1 32个 5*5*3的filter，strides=1  padding=same
#             tf.keras.layers.Flatten(input_shape=(30, 100)),
            # [None, 30 * 100]
            tf.keras.layers.Conv2D(filters=32, input_shape=x_train[0].shape,kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=1, padding="same",
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
                filters=64, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),

    tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, strides=1, padding="same",
        data_format="channels_last", activation=tf.nn.relu
    ),
    tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, strides=1, padding="same",
        data_format="channels_last", activation=tf.nn.relu
    ),
            # 池化层2 2*2窗口  strides=2 [none 8 8 64]
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),
            tf.keras.layers.Dropout(0.25),

            # 全连接层神经网络
            # [none 8 8 64] ----> [None,8*8*64]
            tf.keras.layers.Flatten(),
            # 1024个神经元网络
            tf.keras.layers.Dense(units=2048, activation=tf.nn.relu),
            # 100个神经元网络

            tf.keras.layers.Dense(units=4*63, activation=tf.nn.softmax),

        ])


model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=True),
              # loss='categorical_crossentropy',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])



model.fit(x=x_train, y=y_train,
                       batch_size=32,
                       epochs=500)

model.save_weights('cnn_best.h5')

# model=keras.models.load_model('./ckpt/CNNMnist.h5') model.save('cnn_best.h5')
# model.load_weights('./ckpt/SingleNN.h5')