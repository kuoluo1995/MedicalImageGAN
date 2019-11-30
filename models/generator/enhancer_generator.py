import tensorflow as tf
from models.utils.layers import conv2d, instance_norm2d, deconv2d, res_block2d, leaky_relu


def build_model(batch_input, out_channels, filter_channels=32, coarse_feature_map=None, reuse=False,
                name='enhancer_generator', is_training=False, **kwargs):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        output = tf.nn.relu(
            instance_norm2d(conv2d(batch_input, filter_channels, kernel_size=7, strides=(1, 1), padding='reflect')))
        # down sample
        output = leaky_relu(instance_norm2d(conv2d(output, filter_channels * 2, kernel_size=3, padding='reflect')))
        output = output + coarse_feature_map

        for i in range(3):
            output = res_block2d(output, filter_channels * 2, padding='reflect')

        # up sample
        output = leaky_relu(instance_norm2d(deconv2d(output, filter_channels, kernel_size=3)))

        output = tf.nn.tanh(
            instance_norm2d(conv2d(output, out_channels, kernel_size=7, strides=(1, 1), padding='reflect')))
        return output
