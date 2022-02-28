import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten
# from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import  Model
from tensorflow.python.keras.layers import Input

from tensorflow.python import keras

def main():


    tf.keras.optimizers.SGD()
    model=Sequential([
        Flatten(input_shape=(28,28)),
        Dense(64,activation=tf.nn.sigmoid),
        Dense(128,activation=tf.nn.relu),
        Dense(10,activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy'],
        # loss_weights=None,
        # weighted_metrics=None,
        # run_eagerly=None,
        # steps_per_execution=None,
    )
    print(model)


    # 通过Model建立模型

    data=Input(shape=(784,))
    data2=Dense(64, activation=tf.nn.sigmoid)(data)
    out=Dense(128, activation=tf.nn.softmax)(data2)
    model2=Model(inputs=data,outputs=out)
    print(model2)
    print(model2.layers)
    print(model2.inputs,model2.outputs)

    # 模型结构参数

    print(model2.summary())
    (x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
    print(x_train.shape)






if __name__ == '__main__':
    main()
