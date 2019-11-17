import os
import tensorflow as tf

from data_loader import get_data_loader_by_name
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def _test(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # dataset
        dataset = args['dataset']
        model_dict = args['model']
        data_loader_class = get_data_loader_by_name(dataset['data_loader'])

        test_dict = yaml_utils.read(dataset['test_path'])
        test_data_loader = data_loader_class(False, **test_dict, **args)  # 2d data generator
        # # 3d data generator
        # test_data_loader = data_loader_class(False, base_patch=model_dict['base_patch'], **test_dict, **args)

        model_class = get_model_class_by_name(model_dict['name'])
        model = model_class(data_shape=test_data_loader.get_data_shape(), train_data_loader=None, eval_data_loader=None,
                            test_data_loader=test_data_loader, sess=sess, **args)
        model.test()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = get_config('tumor_loss_2d_pix')
    config['tag'] = 'sigmoid'
    config['is_training'] = False
    config['model']['generator']['name'] = '2d_unet_patch_nf_sigmoid'
    _test(config)
