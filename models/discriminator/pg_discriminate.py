import tensorflow as tf
import numpy as np

from models.utils.layers import down_sampling, pg_conv2d, get_out_channels, minbatch_concat, pg_fully_connect


def build_model(x, process_size, is_transition, alpha_transition, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        if is_transition:
            en_out = down_sampling(x, 1)
            # from output
            en_out = pg_conv2d(en_out, filters=get_out_channels(process_size - 2), kernel_size=1, strides=1,
                               padding='SAME', gain=np.sqrt(2), name='dis_out_{}'.format(en_out.shape[1]))
            en_out = tf.nn.leaky_relu(en_out)
        # from output
        en = pg_conv2d(x, filters=get_out_channels(process_size - 1), kernel_size=1, strides=1, padding='SAME',
                       gain=np.sqrt(2), name='dis_out_{}'.format(x.shape[1]))
        en = tf.nn.leaky_relu(en)

        for i in range(process_size - 1):
            en = pg_conv2d(en, filters=get_out_channels(process_size - 1 - i), kernel_size=3, strides=1, padding='SAME',
                           gain=np.sqrt(2), name='dis_n_1_{}'.format(en.shape[1]))
            en = tf.nn.leaky_relu(en)

            en = pg_conv2d(en, filters=get_out_channels(process_size - 2 - i), kernel_size=3, strides=1, padding='SAME',
                           gain=np.sqrt(2), name='dis_n_2_{}'.format(en.shape[1]))
            en = tf.nn.leaky_relu(en)

            en = down_sampling(en)

            if i == 0 and is_transition:
                en = alpha_transition * en + (1. - alpha_transition) * en_out

        en = minbatch_concat(en)
        en = pg_conv2d(en, filters=get_out_channels(1), kernel_size=3, strides=1, padding='SAME', gain=np.sqrt(2),
                       name='dis_n_1_{}'.format(en.shape[1]))
        en = tf.nn.leaky_relu(en)

        en = pg_conv2d(en, filters=get_out_channels(1), kernel_size=4, strides=1, padding='VALID', gain=np.sqrt(2),
                       name='dis_n_2_{}'.format(en.shape[1]))
        en = tf.nn.leaky_relu(en)

        en = tf.layers.flatten(en)
        # for D
        output = pg_fully_connect(en, 1, gain=1, name='dis_n_dense')
        return output  # tf.nn.sigmoid(output),
