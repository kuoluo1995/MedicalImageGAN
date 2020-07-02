import tensorflow as tf
import numpy as np
from models.utils.layers import down_sampling, pix_norm, pg_conv2d, get_out_channels, upscale


def build_model(x, process_size, is_transition, alpha_transition, reuse=False, name='decode', **kwargs):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        xs = list()
        for i in range(process_size + 1):
            xs.append(x)
            x = down_sampling(x, 1)
        xs.reverse()
        de = pix_norm(x)
        de = pg_conv2d(de, filters=get_out_channels(1), kernel_size=4, strides=1, padding='OTHER', gain=np.sqrt(2) / 4,
                       name='gen_n_1_{}'.format(de.shape[1]))
        de = tf.nn.leaky_relu(de)
        de = pix_norm(de)
        de = tf.concat([de, xs[1]], 3)
        de = pg_conv2d(de, filters=get_out_channels(1), kernel_size=3, strides=1, padding='SAME', gain=np.sqrt(2),
                       name='gen_n_2_{}'.format(de.shape[1]))
        de = tf.nn.leaky_relu(de)
        de = pix_norm(de)
        for i in range(process_size - 1):
            if i == process_size - 2 and is_transition:
                # to output
                de_out = pg_conv2d(de, filters=1, kernel_size=1, strides=1, padding='SAME', gain=np.sqrt(2),
                                   name='gen_out_{}'.format(de.shape[1]))
                de_out = upscale(de_out, 2)
            de = upscale(de, 2)
            de = tf.concat([de, xs[i + 2]], 3)
            de = pg_conv2d(de, filters=get_out_channels(i + 1), kernel_size=3, strides=1, padding='SAME',
                           gain=np.sqrt(2), name='gen_n_1_{}'.format(de.shape[1]))
            de = tf.nn.leaky_relu(de)
            de = pix_norm(de)

            de = pg_conv2d(de, filters=get_out_channels(i + 1), kernel_size=3, strides=1, padding='SAME',
                           gain=np.sqrt(2), name='gen_n_2_{}'.format(de.shape[1]))
            de = tf.nn.leaky_relu(de)
            de = pix_norm(de)
        # to out
        de = pg_conv2d(de, filters=1, kernel_size=1, strides=1, padding='SAME', gain=1,
                       name='gen_out_{}'.format(de.shape[1]))
        if process_size == 1:
            return de
        if is_transition:
            de = (1 - alpha_transition) * de_out + alpha_transition * de
        return de
