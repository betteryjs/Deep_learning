from tensorflow.python.keras.applications.vgg16 import VGG16,decode_predictions, preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


# tf.keras.applications.VGG16()









def predict():
    model = VGG16()
    print(model.summary())
    # 预测一张图类别
    # 加载图片并输入到模型
    # 224 224是vgg的输入要求
    image=load_img("../images/bus.jpg",target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    print(image.shape)
    # 预测之前做图片的数据处理
    image=preprocess_input(image)
    y_predictions=model.predict(image)
    print(y_predictions)
    label=decode_predictions(y_predictions)
    print(label)









if __name__ == '__main__':
    predict()