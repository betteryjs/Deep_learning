import io
import tensorflow as tf
import grpc
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.saved_model import signature_constants
from utils.ssd_utils import BBoxUtility
from utils.tag_img import tag_picture


def make_prediction(image):
    """

    处理前端传入的参数进行预测
    :return:
    """

    def resize_img(image,input_size):
        img=io.BytesIO()
        img.write(image)
        rgb=Image.open(img).convert("RGB")
        if input_size:
            rgb=rgb.resize((input_size[0],input_size[1]))
        return rgb


    resize=resize_img(image,(300,300))
    image_array=img_to_array(resize)
    # 3---->4
    image_tensor=preprocess_input(np.array([image_array]))
    print(image_tensor.shape)
    # 8500:grpc 8501:http
    with grpc.insecure_channel("0.0.0.0:8500") as channel:
        # stub通道预测
        stub=prediction_service_pb2_grpc.PredictionServiceStub(channel)
        # 构造 tensorflow serving 请求格式
        request=predict_pb2.PredictRequest()
        request.model_spec.name="commodity"
        # 默认签名
        request.model_spec.signature_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        request.inputs["images"].CopyFrom(tf.contrib.util.make_tensor_proto(image_tensor,shape=[1,300,300,3]))
        # 模型服务的requests
        response=stub.Predict(request)




        # 会话去解析模型服务返回的结果

        with tf.Session() as sess:
            _res=sess.run(tf.convert_to_tensor(response.outputs["concat_3:0"]))
            bbox= BBoxUtility(9)
            y_predict=bbox.detection_out(_res)






    return tag_picture(image_array,y_predict)



if __name__ == '__main__':
    pass