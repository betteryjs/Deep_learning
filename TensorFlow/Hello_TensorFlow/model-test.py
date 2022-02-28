import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('./',one_hot=True)
# 创建变量和占位符
X=tf.compat.v1.placeholder(dtype=tf.float64,shape=[None,784])
y=tf.compat.v1.placeholder(dtype=tf.float64,shape=[None,10])


# 卷积核在神经网络中是变量
# 变量生成方法
def gen_v(shape):
        return tf.Variable(
            initial_value=tf.random.normal(dtype=tf.float64,shape=shape,stddev=0.1),
            dtype=tf.float64)
# 定义方法，完成卷积操作

def conv(input_data,filters_):
    return tf.nn.conv2d(input=input_data,filters=filters_,strides=[1,1,1,1],padding="SAME")

# 定义池化操作
def pool(input_data):
    return tf.nn.max_pool(input=input_data,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# 第一层卷积
input_data1=tf.reshape(X,shape=[-1,28,28,1])
# 卷积核
filter1=gen_v(shape=[3,3,1,64])
# 偏差 bias
b1=gen_v(shape=[64])
conv1=conv(input_data1,filter1)+b1
# 池化
pool1=pool(conv1)
# 激活 函数
active1=tf.nn.relu(pool1)


# 第二层卷积

# 使用的是第一层卷积的数据

filter2=gen_v(shape=[3,3,64,64])

b2=gen_v(shape=[64])

conv2=conv(active1,filter2)+b2

# 池化

pool2=pool(conv2)

active2=tf.nn.sigmoid(pool2)
# 全连接层
# 1024个连接 1024个方程 1024个神经元

fc_w=gen_v(shape=[7*7*64,1024])
fc_b=gen_v(shape=[1024])

conn=tf.matmul(tf.reshape(active2,shape=[-1,7*7*64]),fc_w)+fc_b

# dropout 仿止过拟合
kp=tf.compat.v1.placeholder(dtype=tf.float64,shape=None)

dropout_=tf.nn.dropout(conn,rate=kp)


# 输出层
# 10个类别 0 ~ 9
out_w=gen_v(shape=[1024,10])

out_b=gen_v(shape=[10])

out=tf.matmul(dropout_,out_w)+out_b


prob=tf.nn.softmax(out)

cost=tf.reduce_mean(tf.reduce_sum(y*tf.math.log(1/prob),axis=-1))


adam=tf.compat.v1.train.AdamOptimizer()

optimizer=adam.minimize(cost)


saver=tf.compat.v1.train.Saver()

# gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config_ = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True, allow_soft_placement=True)

with tf.compat.v1.Session(config=config_) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(10):
        c = 0
        for j in range(100):
            X_train, y_train = mnist.train.next_batch(550)
            optimizer_, cost_ = sess.run(fetches=[optimizer, cost],
                                         feed_dict={X: X_train, y: y_train, kp: 0.5})

            c += cost_ / 100
            print('里层循环次数：%d，每次损失：%0.4f' % (j, cost_))

        print('--------------执行次数：%d损失函数是：%0.4f----------------' % (i, c))
        saver.save(sess, save_path='./cnn/model', global_step=i)

def test():
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, './cnn/model-9')

        X_test, y_test = mnist.test.next_batch(2000)

        prob_ = sess.run(prob, feed_dict={X: X_test, kp: 1.0})

        y_test = y_test.argmax(axis=-1)

        prob_ = prob_.argmax(axis=-1)

        print((y_test == prob_).mean())