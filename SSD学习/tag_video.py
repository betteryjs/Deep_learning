"""
配置获取相关预测数据类别，网络参数
获取摄像头视频
获取摄像每帧数据，进行格式形状处理
模型预测、结果NMS过滤
画图：显示物体位置，FPS值（每秒帧数）
"""
import cv2
import numpy as np
import seaborn as sns
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from utils.ssd_utils import BBoxUtility
from nets.ssd_net import SSD300




class VideoTag(object):
    def __init__(self,model,input_shape):
        self.input_shape=input_shape
        self.class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                            "tvmonitor"]
        self.model=model
        self.class_nums=len(self.class_names)
        self.bbox_util=BBoxUtility(self.class_nums)
        self.conf_thresh=0.8 # 检测图片的阈值
        self.class_colors=sns.color_palette('hls',self.class_nums-1)


    def run(self,file_path):
        """

        运行摄像头 捕捉每一帧数据

        :return:
        """
        # 获取摄像头视频

        cap=cv2.VideoCapture(file_path)

        if not cap.isOpened() :
            raise IOError("open local files error!!")

        while True:
            retval,orig_images=cap.read()
            if not retval:
                print("视频检测结束")
            source_images=np.copy(orig_images)
            # 修改每一帧的数据及图片的格式BGR ---> RGB
            im_size=(self.input_shape[0],self.input_shape[1])
            resized=cv2.resize(orig_images,im_size)

            rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
            # print(source_images.shape) # (480, 640, 3)
            to_draw=cv2.resize(resized,(int(source_images.shape[1]),int(source_images.shape[0])))

            # 获取摄像每帧数据，进行格式形状处理
            inputs=[img_to_array(rgb)]
            x=preprocess_input(np.array(inputs))
            y=self.model.predict(x)
            results=self.bbox_util.detection_out(y)
            print(results[0].shape)

            if len(results) > 0 and len(results[0]) > 0:
                # 获取每个框的位置以及类别概率
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                # 过滤概率小的
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.conf_thresh]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                for i in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                    ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                    xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                    ymax = int(round(top_ymax[i] * to_draw.shape[0]))

                    # 对于四个坐标物体框进行画图显示
                    class_num = int(top_label_indices[i])
                    cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax),
                                  [i * 255 for i in self.class_colors[class_num]], 2)
                    text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i])

                    # 文本框进行设置显示
                    text_top = (xmin, ymin - 10)
                    text_bot = (xmin + 80, ymin + 5)
                    text_pos = (xmin + 8, ymin)


                    cv2.rectangle(to_draw, text_top, text_bot, [i * 255 for i in self.class_colors[class_num]], -1)
                    cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            # 模型预测、结果NMS过滤

            # 画图：显示物体位置，FPS值（每秒帧数）

            # 计算 FPS显示
            fps = "FPS: " + str(cap.get(cv2.CAP_PROP_FPS))

            # 画出FPS
            cv2.rectangle(to_draw, (0, 0), (50, 17), (255, 255, 255), -1)
            cv2.putText(to_draw, fps, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)


        # 显示当前图片

            cv2.imshow("SSD detector result",to_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
                break

        # 释放capture资源
        cap.release()
        cv2.destroyAllWindows()


        return None

if __name__ == '__main__':
    input_shape=(300,300,3)
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                   "tvmonitor"]
    model=SSD300(input_shape,num_classes=len(class_names))

    # 加载已经训练好的模型
    model.load_weights('./ckpt/pre_trained/weights_SSD300.hdf5')
    videotag=VideoTag(model=model,input_shape=input_shape)
    videotag.run(0)