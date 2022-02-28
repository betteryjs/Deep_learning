from tensorflow import keras
import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

class CNNMnist(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()
        print(self.x_train.shape)
        print(self.y_train.shape)
        # 进行数据的归一化
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        self.model = tf.keras.models.Sequential([
            # 卷积层1 32个 5*5*3的filter，strides=1  padding=same
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=5, strides=1, padding="same",input_shape=[28,28,1],
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
                filters=64, kernel_size=5, strides=1, padding="same",
                data_format="channels_last", activation=tf.nn.relu
            ),

            # 池化层2 2*2窗口  strides=2 [none 8 8 64]
            tf.keras.layers.MaxPool2D(
                pool_size=2,
                strides=2,
                padding='same',
            ),

            # 全连接层神经网络
            # [none 8 8 64] ----> [None,8*8*64]
            tf.keras.layers.Flatten(),
            # 1024个神经元网络
            tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
            # 100个神经元网络
            tf.keras.layers.Dense(units=100, activation=tf.nn.softmax),

        ])

    def get_model(self):
        return self.model

    def cnn_compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                            # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            loss=tf.keras.losses.sparse_categorical_crossentropy,
                            metrics=["accuracy"])
        return None

    def fit(self):
        check = tf.keras.callbacks.ModelCheckpoint(
            # filepath='./ckpt/model-{epoch:02d}.hdf5',
            filepath='./ckpt/model-{epoch:02d}-{val_accuracy:6f}.h5',
            monitor='val_accuracy',
            save_best_only=True, # 是否保存比上次好的模型
            save_weights_only=True,
            mode='auto',
            verbose=1,
            period=1

        )
        self.model.fit(x=self.x_train, y=self.y_train,
                       batch_size=32,
                       callbacks=[check],
                       epochs=50)
        self.model.save("ckpt/CNNMnist.h5")
        return None

    def evaluate(self):

        # if os.path.exists('./ckpt/model-46-0.974600.h5'):
        #
        #     load_model=self.model.load_weights('./ckpt/model-46-0.974600.h5')

        loss,acc=self.model.evaluate(self.x_test,self.y_test,verbose=2)
        # predictions=self.model.predict(self.x_test)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        return loss,acc

        # test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        #
        # print(test_loss, test_acc)
        # return None
    def model_evaluate(self):

        # if os.path.exists('./ckpt/model-46-0.974600.h5'):
        #
        #     load_model=self.model.load_weights('./ckpt/model-46-0.974600.h5')
        model=keras.models.load_model('./ckpt/CNNMnist.h5')
        loss,acc=model.evaluate(self.x_test,self.y_test,verbose=2)
        # predictions=self.model.predict(self.x_test)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
        return loss,acc


if __name__ == '__main__':
    cnn = CNNMnist()
    cnn.cnn_compile()
    # cnn.fit()
    # print(cnn.model.summary())


    cnn.model_evaluate()
    # cnn.evaluate()

    # cnn.model.load_weights('./ckpt/SingleNN.h5')
    # reconstructed_model = cnn.model.load_weights("./ckpt/model-46-0.974600.h5")
    # loss,acc=reconstructed_model.evaluate(cnn.x_test, cnn.y_test, verbose=2)
    # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    # predictions=cnn.evaluate()
    #
    # predictions=tf.argmax(predictions,axis=-1)
    # y_test_true=tf.argmax(cnn.y_test,axis=-1)
    # print(tf.cast(predictions==y_test_true,dtype=tf.int32).mean())
