import tensorflow as tf
import os
import warnings

warnings.filterwarnings("ignore")
tf.compat.v1.disable_eager_execution()

# a = tf.constant(11.0)
# b = tf.constant(12.0)
#
# c = tf.add(a, b)
#
# # 获取默认图
#
# g = tf.compat.v1.get_default_graph()
# print("获取当前加法运算的图", g)
#
# # 打印所有操作的图
#
# print(a.graph)
# print(b.graph)
# print(c.graph)
#
# # 创建另外一张图
#
# new_g = tf.Graph()
#
# with new_g.as_default():
#     new_a = tf.constant(11.0)
#     new_b = tf.constant(12.0)
#
#     new_c = tf.add(new_a, new_b)
#
# # 打印所有操作的图
#
# print(new_a.graph)
# print(new_b.graph)
# print(new_c.graph)
#
#
#
# with tf.compat.v1.Session() as sess:
#     res = sess.run(c)
#     print(res)



a = tf.constant(11.0)
b = tf.constant(12.0)

c = tf.add(a, b)

# 获取默认图

g = tf.compat.v1.get_default_graph()
print("获取当前加法运算的图", g)

# 打印所有操作的图
print(a.graph)
print(b.graph)
print(c.graph)

config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True)
with tf.compat.v1.Session(config=config) as sess:
    res = sess.run(c)

    filewriter=tf.compat.v1.summary.FileWriter("./summary/",graph=sess.graph)


    print(res)
