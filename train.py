import os
import tensorflow as tf

from data_loader import get_data_loader_by_name
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def train(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # dataset
        dataset = args['dataset']
        data_loader_class = get_data_loader_by_name(dataset['data_loader'])

        train_dict = yaml_utils.read(dataset['train_path'])
        train_data_loader = data_loader_class(True, **train_dict, **dataset)  # 2d data generator
        # train_data_loader = data_loader_class(True, base_patch=model_dict['base_patch'], **train_dict,
        #                                       **args)  # 3d data generator

        eval_dict = yaml_utils.read(dataset['eval_path'])
        eval_data_loader = data_loader_class(False, **eval_dict, **dataset)  # 2d data generator
        # eval_data_loader = data_loader_class(False, base_patch=model_dict['base_patch'], **eval_dict,
        #                                      **args)  # 3d data generator
        model_dict = args['model']
        model_class = get_model_class_by_name(model_dict['name'])
        model = model_class(data_shape=train_data_loader.get_data_shape(),
                            batch_size=train_data_loader.get_batch_size(),
                            in_channels=train_data_loader.get_in_channels(),
                            out_channels=train_data_loader.get_out_channels(),
                            train_data_loader=train_data_loader,
                            eval_data_loader=eval_data_loader, test_data_loader=None, sess=sess, **args)
        model.train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # config = get_config('patch_2d_pix')
    # config['tag'] = 'bass'
    config = get_config('patch_2d_pix')
    config['tag'] = 'channel3'
    config['dataset']['in_channels'] = 3
    config['model']['name'] = 'pix2pix_gan_channels'
    # config = get_config('edge_2d_pix')
    # config['tag'] = 'edge'

    # config = get_config('sobel_2d_pix')
    # config['tag'] = 'base'

    # config = get_config('ssim_2d_pix')
    # config['tag'] = 'base'

    # config = get_config('ssim_2d_pix')
    # config['tag'] = 'ssim2'

    # config = get_config('ssim_soble_2d_pix')
    # config['tag'] = 'sobel'

    # config = get_config('ssim_edge_2d_pix')
    # config['tag'] = 'base'
    train(config)
