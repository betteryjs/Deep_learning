import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import load_img
import numpy as np
tf.compat.v1.disable_eager_execution()
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)


class TransferModel(object):
    def __init__(self):
        # rescale 标准化
        self.train_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        self.test_generator = ImageDataGenerator(rescale=1.0 / 255.0)
        # 指定训练数据和测试数据的目录
        self.train_dir = 'data/train'
        self.test_dir = 'data/test'
        # 定义网络训练相关参数
        self.image_size = (224, 224)
        self.batch_size = 32
        # 定义迁移学习的基类模型
        # 不包含VGG全连接层 vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
        self.base_model = VGG16(weights='imagenet', include_top=False)

        self.label_dict = {
            '0': 'bus',
            '1': 'dinosaurs',
            '2': 'elephants',
            '3': 'flowers',
            '4': 'horse'
        }

    def get_local_data(self):
        """
        读取本地的图片数据以及类别

        :return: test data and train data
        """

        train_gen = self.train_generator.flow_from_directory(
            directory=self.train_dir,
            target_size=self.image_size,
            class_mode="binary",
            batch_size=self.batch_size,
            shuffle=True

        )
        test_gen = self.test_generator.flow_from_directory(
            directory=self.test_dir,
            target_size=self.image_size,
            class_mode="binary",
            batch_size=self.batch_size,
            shuffle=True
        )
        return train_gen, test_gen

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
        y_perdict = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)

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

    def fit_generator(self, model, train_gen, test_gen):
        """

        训练模型
        :return:
        """
        modelckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath='./ckpt/transfer_{epoch:02d}-{val_accuracy:4f}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1

        )
        model.fit_generator(
            generator=train_gen,
            epochs=20,
            validation_data=test_gen,
            callbacks=[modelckpt]

        )

        return None

    def predict(self, model):
        """

         预测类别
        :return:
        """
        model.load_weights(r'.\ckpt\transfer_09-0.960000.h5')
        image = load_img("./data/test/dinosaurs/400.jpg", target_size=(224, 224))
        image = img_to_array(image)
        # image.shape ---> [224,224,3]
        # 转换成四维函数 [224,224,3] ----> [1,224,224,3]
        # tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
        # image=tf.reshape(image,shape=[1, 224, 224, 3])
        image=np.array(image).reshape([1, image.shape[0], image.shape[1],image.shape[2]])
        print(image.shape)

        image = preprocess_input(image)



        predictions= model.predict(image)
        res=np.argmax(predictions,axis=1)[0]
        print(res)

        print(self.label_dict[str(res)])

if __name__ == '__main__':
    # 训练  (32,224,224,3)
    tm = TransferModel()
    train_gen,test_gen=tm.get_local_data()
    print(train_gen)

    print(tm.base_model.summary())
    transfer_model=tm.refine_base_model()
    tm.freeze_model()
    tm.model_compile(transfer_model)
    tm.fit_generator(transfer_model,train_gen,test_gen)

    # # 预测
    # tm = TransferModel()
    # model = tm.refine_base_model()
    # tm.predict(model)
