import os
import tensorflow as tf
from models import get_model_class_by_name
from utils import yaml_utils
from utils.dict_object_utils import Option


def get_config(name):
    base_config = yaml_utils.read('configs/base.yaml')
    base_config.update(yaml_utils.read('configs/' + name + '.yaml'))
    base_config['name'] = name
    tf.set_random_seed(19)
    return base_config


def train(option, config):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        model_class = get_model_class_by_name(option.model.name)
        model = model_class(sess=sess, **config)
        model.build_model()
        model.train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = get_config('nf')
    option = Option(config)
    train(option, config)
