import os
import tensorflow as tf
from pathlib import Path

from data_loader import get_data_loader_by_name
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def train(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    # dataset
    dataset = args['dataset']
    data_loader_class = get_data_loader_by_name(dataset['data_loader'])
    train_dict = yaml_utils.read(dataset['train_path'])
    train_data_loader = data_loader_class(True, **train_dict, **dataset)  # 2d data generator
    eval_dict = yaml_utils.read(dataset['eval_path'])
    eval_data_loader = data_loader_class(False, **eval_dict, **dataset)  # 2d data generator

    # model
    model_dict = args['model']
    process_sizes = args['process_sizes']
    checkpoint_dir = Path(model_dict['checkpoint_dir']) / dataset['name'] / model_dict['name'] / args['tag']
    log_dir = Path(model_dict['log_dir']) / dataset['name'] / model_dict['name'] / args['tag']

    for i in range(1, len(process_sizes)):
        is_transition = True if i % 2 == 0 else False
        output_size = [2 ** (process_sizes[i] + 1), 2 ** (process_sizes[i] + 1)]  # 4,8,16,32,64,128,256
        save_path = checkpoint_dir / str(process_sizes[i])
        save_path.mkdir(parents=True, exist_ok=True)
        read_path = checkpoint_dir / str(process_sizes[i - 1])
        log_path = log_dir / '{}_{}x{}'.format(i, *output_size)
        log_path.mkdir(parents=True, exist_ok=True)
        with tf.Session(config=tf_config) as sess:
            model_class = get_model_class_by_name(model_dict['name'])
            model = model_class(data_shape=train_data_loader.get_data_shape(),
                                train_data_loader=train_data_loader, eval_data_loader=eval_data_loader,
                                test_data_loader=None, is_transition=is_transition, output_size=output_size,
                                process_size=process_sizes[i], save_path=str(save_path), read_path=str(read_path),
                                log_path=str(log_path), model_id=i, max_id=len(process_sizes), sess=sess, **args)
            model.train(tf_config)
        tf.reset_default_graph()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = get_config('pg_pix2d')
    config['tag'] = 'base'
    train(config)
