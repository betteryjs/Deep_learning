from flask import Flask, jsonify
from flask import request, abort, send_file, render_template
import base64
from urllib import parse

from prediction import make_prediction, make_prediction_v2

VOC_LABELS = {
    '0': 'Background',
    '1': 'clothes',
    '2': 'pants',
    '3': 'shoes',
    '4': 'watch',
    '5': 'phone',
    '6': 'audio',
    '7': 'computer',
    '8': 'books'
}


app = Flask(__name__)


@app.route("/api/v3/prediction/commodity", methods=['POST'])
def commodity_predict_v2():
    """
    对接百度机器人的接口逻辑实现
    :return:
    """
    # 1、接到请求数据，然后进行数据解析
    req = request.get_json()
    requestId = req['requestId']
    image = req['image']
    # 编码处理
    img_str = parse.unquote(image)
    # base64的解码,之后是图片本身的内容
    img = base64.b64decode(img_str.encode())

    # 2、定义一个新的预测接口，返回预测结果
    # y_predict (100, 6)
    y_predict = make_prediction_v2(img)
    # y_predict[0][:, 1].shape[0] = y_predict[0].shape[0] 预测的物体个数

    # 3、得到预测结果，按照百度物体检测协议说明，返回指定格式
    # {
    #     "score": 0.996184, "root": "商品-电脑办公", "keyword": "台式机"
    # }
    resp = {
        "result": []
    }
    for i in range(y_predict[0][:, 1].shape[0]):
        resp['result'].append({"score": y_predict[0][:, 1][i], "root":" ",
                               "keyword": VOC_LABELS[str(y_predict[0][:, 0][i])]})
    resp['extInfos'] = {}
    resp['filterThreshold'] = 0.7
    resp['resultNum'] = y_predict[0][:, 1].shape[0]
    resp['requestId'] = requestId

    return jsonify(resp)


@app.route("/api/v1/prediction/commodity", methods=['POST'])
def commodity_predict():
    """
    商品图片预测REST接口
    """
    # 获取用户上传图片
    image = request.files.get('image')
    if not image:
        abort(400)
    # 预测标记
    result_img = make_prediction(image.read())
    data = result_img.read()
    result_img.close()

    return data, 200, {'Content-Type': 'image/png'}


@app.route("/")
def index():
    """
    Web页面
    """
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
