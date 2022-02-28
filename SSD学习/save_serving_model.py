import tensorflow as tf
import os
from nets.ssd_net import SSD300
from keras import backend as K



def export_serving_model( version=1, path='./serving_model/commodity'):
    """

    导出模型到pd文件
    :return:
    """
    model_path=os.path.join(
        tf.compat.as_bytes(path),
        tf.compat.as_bytes(str(version))
    )

    model = SSD300((300, 300, 3), num_classes=9)
    model.load_weights('./ckpt/fine_tuning/weights.00-4.50.hdf5', by_name=True)




    # 3. tf.saved_model.simple_save 导出


    with K.get_session() as sess:
        tf.saved_model.simple_save(
            session=sess,
            export_dir=model_path,
            inputs={"images":model.input}, #
            outputs={model.output.name:model.output},
            legacy_init_op=None

        )




if __name__ == '__main__':
    export_serving_model()
