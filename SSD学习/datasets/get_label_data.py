import pickle
from xml.etree import ElementTree as ET
import numpy as np
import os


class XmlProcess(object):
    def __init__(self, file_path):
        self.xml_path = file_path
        self.num_classes = 8
        self.data=dict()

    def on_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        if name == 'clothes':
            one_hot_vector[0] = 1
        elif name == 'pants':
            one_hot_vector[1] = 1
        elif name == 'shoes':
            one_hot_vector[2] = 1
        elif name == 'watch':
            one_hot_vector[3] = 1
        elif name == 'phone':
            one_hot_vector[4] = 1
        elif name == 'audio':
            one_hot_vector[5] = 1
        elif name == 'computer':
            one_hot_vector[6] = 1
        elif name == 'books':
            one_hot_vector[7] = 1
        else:
            print('unknown label: %s' % name)
        return one_hot_vector

    def process_xml(self):
        """

         # 处理图片的标注信息 解析图片大小
         图片中所有物体的位置 类别 存入pkl文件中
        :return:
        """

        for filename in os.listdir(self.xml_path):

            et = ET.parse(os.path.join(self.xml_path, filename))
            root = et.getroot()
            # print(root)
            size = root.find("size")
            width = float (size.find("width").text)
            height = float( size.find("height").text)
            depth = size.find("depth").text
            print(width, height, depth)

            bounding_boxs = []
            one_hot_classes = []
            for object_tree in root.findall("object"):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text) / width
                    ymin = float(bounding_box.find('ymin').text) / height
                    xmax = float(bounding_box.find('xmax').text) / width
                    ymax = float(bounding_box.find('ymax').text) / height
                bounding_boxs.append([xmin, ymin, xmax, ymax])
                class_name = object_tree.find('name').text
                # 进行one_hot 编码

                one_hot_class = self.on_hot(class_name)
                one_hot_classes.append(one_hot_class)
            # 进行物体位置和目标值的one_hot编码进行拼接
            image_name=root.find("filename").text
            bounding_boxs = np.array(bounding_boxs)
            one_hot_classes = np.array(one_hot_classes)
            image_data=np.hstack((bounding_boxs,one_hot_classes))
            print(image_data)
            self.data[image_name]=image_data












        return None


if __name__ == '__main__':
    Annotations = "D:\my_projects\Data_Analysis_Project\Deep_learning_and_machine_vision\SSD学习\datasets\commodity\Annotations"
    xp = XmlProcess(Annotations)
    xp.process_xml()
    # print(xp.data)
    pickle.dump(xp.data, open('commodity_gt.pkl', 'wb'))
