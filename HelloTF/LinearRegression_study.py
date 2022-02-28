import os.path

import tensorflow as tf

tf.compat.v1.disable_eager_execution()




# 定义命令行参数

tf.compat.v1.flags.DEFINE_integer("maxStep",1000,"train_step_number")


FLAGS=tf.compat.v1.flags.FLAGS


def LinearRegression():
    with tf.compat.v1.variable_scope("originData"):
        X = tf.random.normal(dtype=tf.float64, shape=[100, 1], stddev=1.0, name="origin_data_X")
        y_true = tf.matmul(X, [[0.8]]) + 0.7

    with tf.compat.v1.variable_scope("linearModel"):
        weight = tf.Variable(
            initial_value=tf.random.normal(dtype=tf.float64, shape=[1, 1]),
            dtype=tf.float64, name="w"
        )

        bias = tf.Variable(
            initial_value=tf.random.normal(dtype=tf.float64, shape=[1]),
            dtype=tf.float64, name="b"
        )
        y_prerdict = tf.matmul(X, weight) + bias

    with tf.compat.v1.variable_scope("loss"):
        # 损失函数

        loss = tf.reduce_mean(tf.square(y_true - y_prerdict))
    with tf.compat.v1.variable_scope("optimizer"):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 收集变量

    tf.compat.v1.summary.scalar("loss", loss)
    tf.compat.v1.summary.histogram("weight", weight)
    tf.compat.v1.summary.histogram("bias", bias)

    # 合并变量

    merge = tf.compat.v1.summary.merge_all()

    # 创建saver
    saver = tf.compat.v1.train.Saver()

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config_ = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True, allow_soft_placement=True)
    with tf.compat.v1.Session(config=config_) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        filewriter = tf.compat.v1.summary.FileWriter("summary", graph=sess.graph)

        for i in range(FLAGS.maxStep):
            sess.run(fetches=optimizer)
            print("step : {} loss: {} weight: {} bias: {}".format(i,sess.run(loss),sess.run(weight),sess.run(bias)))

            summary = sess.run(merge)
            filewriter.add_summary(summary, i)
            if i % 100 == 0:
                # chick point
                path=os.path.join("ckpt","linerRegression")

                saver.save(sess, path)

        # 读取saver
        # saver.restore(sess,"ckpt_path")


if __name__ == '__main__':
    LinearRegression()
