import tensorflow as tf
from tensorflow import keras
import os

class SingleNN(object):


    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist

        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        self.x_train = self.x_train / 255.0

        self.x_test = self.x_test / 255.0
        self.model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
    def singleCompile(self):
        """
        编辑模型优化器 损失 准确率

        :return:
        """
        # 优化器
        # 损失函数

        self.model.compile(
                      optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])



    def singleFit(self):
        """
        进行fit训练
        :return:
        """
        # 训练样本的特征值和目标值

        # SingleNN.model.fit(
        #     x=self.x_train,
        #     y=self.y_train,
        #     epochs=20 # 训练多少遍
        #
        #
        # )

        check=keras.callbacks.ModelCheckpoint(
            # filepath='./ckpt/model-{epoch:02d}.hdf5',
            filepath='./ckpt/model-{epoch:02d}-{accuracy:.2f}.hdf5',
            monitor='accuracy',
            save_best_only=True, # 是否保存比上次好的模型
            save_weights_only=True,
            mode='max',
            verbose = 1,

        )

        board=keras.callbacks.TensorBoard(
            log_dir='./summary/',
            histogram_freq=1,
            write_images=True,
            write_graph=True,

        )

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=10,
            callbacks=[check,board]
        )


    def singleEvalatue(self):
        # 评估模型测试效果
        test_loss,test_score=self.model.evaluate(self.x_test,self.y_test)
        print(test_loss,test_score)

    def singlePredict(self):
        if os.path.exists('./ckpt/SingleNN.h5'):

            self.model.load_weights('./ckpt/SingleNN.h5')

        predictions=self.model.predict(self.x_test)
        return predictions


if __name__ == '__main__':
    snn=SingleNN()
    snn.singleCompile()
    snn.singleFit()
    snn.singleEvalatue()
    # snn.model.save_weights("ckpt/SingleNN.h5")
    # predictions=snn.singlePredict()
    # print(predictions)
    # predictions=tf.argmax(predictions,axis=-1)
    # print(predictions)

