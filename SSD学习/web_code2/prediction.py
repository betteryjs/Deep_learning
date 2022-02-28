from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.saved_model import signature_constants
from utils.ssd_utils import BBoxUtility
from utils.tag_img import tag_picture
import tensorflow as tf
from PIL import Image
import grpc
import numpy as np
import io


def make_prediction(image):
    """
    对于前端传入的参数进行预测
    :return:
    """
    # - 1、获取读取后台读取的图片
    def resize_img(image, input_size):
        img = io.BytesIO()
        img.write(image)
        # 使用pillow image 接收这个图片
        rgb = Image.open(img).convert('RGB')

        # 换砖大小
        if input_size:
            rgb = rgb.resize((input_size[0], input_size[1]))

        return rgb

    # - 2、图片大小处理，转换数组
    resize = resize_img(image, (300, 300))
    image_array = img_to_array(resize)
    # 3---> 4维度
    image_tensor = preprocess_input(np.array([image_array]))

    # - 3、打开通道channel, 构建stub，预测结果， predict_pb2进行预测请求创建
    # 8500:grpc 8501:http
    with grpc.insecure_channel("127.0.0.1:8500") as channel:
        # stub通道
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # 构造请求 tensorflow serving请求格式
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'commodity'
        # 默认签名即可
        request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image_tensor, shape=[1, 300, 300, 3]))

        # 与模型服务进行请求获取预测结果
        # 模型服务的request {'concat_3:0': tensor类}
        results = stub.Predict(request)

        # 会话去解析模型服务返回的结果
        with tf.Session() as sess:

            _res = sess.run(tf.convert_to_tensor(results.outputs['concat_3:0']))

            # 进行预测结果的NMS过滤
            # 物体检测的类别数量8 + 1
            bbox = BBoxUtility(9)
            y_predict = bbox.detection_out(_res)

    return tag_picture(image_array, y_predict)


def make_prediction_v2(image):
    """
    提供给百度机器人的预测接口
    :return:
    """
    # - 1、获取读取后台读取的图片
    def resize_img(image, input_size):
        img = io.BytesIO()
        img.write(image)
        # 使用pillow image 接收这个图片
        rgb = Image.open(img).convert('RGB')

        # 换砖大小
        if input_size:
            rgb = rgb.resize((input_size[0], input_size[1]))

        return rgb

    # - 2、图片大小处理，转换数组
    resize = resize_img(image, (300, 300))
    image_array = img_to_array(resize)
    # 3---> 4维度
    image_tensor = preprocess_input(np.array([image_array]))

    # - 3、打开通道channel, 构建stub，预测结果， predict_pb2进行预测请求创建
    # 8500:grpc 8501:http
    with grpc.insecure_channel("127.0.0.1:8500") as channel:
        # stub通道
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # 构造请求 tensorflow serving请求格式
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'commodity'
        # 默认签名即可
        request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image_tensor, shape=[1, 300, 300, 3]))

        # 与模型服务进行请求获取预测结果
        # 模型服务的request {'concat_3:0': tensor类}
        results = stub.Predict(request)

        # 会话去解析模型服务返回的结果
        with tf.Session() as sess:

            _res = sess.run(tf.convert_to_tensor(results.outputs['concat_3:0']))

            # 进行预测结果的NMS过滤
            # 物体检测的类别数量8 + 1
            bbox = BBoxUtility(9)
            y_predict = bbox.detection_out(_res)

    return y_predict