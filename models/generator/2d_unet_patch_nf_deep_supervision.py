import tensorflow as tf
from models.utils.layers import conv2d, instance_norm2d, leaky_relu, deconv2d


def build_model(image, out_channels, filter_channels=64, reuse=False, name='unet', is_training=False, **kwargs):
    dropout_rate = 0.5 if is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        e1 = instance_norm2d(conv2d(image, filter_channels, stride=1, name='g_conv_e1'), 'g_in_e1')
        e2 = instance_norm2d(conv2d(leaky_relu(e1), filter_channels * 2, name='g_conv_e2'), 'g_in_e2')
        e3 = instance_norm2d(conv2d(leaky_relu(e2), filter_channels * 4, name='g_conv_e3'), 'g_in_e3')
        e4 = instance_norm2d(conv2d(leaky_relu(e3), filter_channels * 8, name='g_conv_e4'), 'g_in_e4')
        e5 = instance_norm2d(conv2d(leaky_relu(e4), filter_channels * 8, name='g_conv_e5'), 'g_in_e5')
        e6 = instance_norm2d(conv2d(leaky_relu(e5), filter_channels * 8, name='g_conv_e6'), 'g_in_e6')
        e7 = instance_norm2d(conv2d(leaky_relu(e6), filter_channels * 8, name='g_conv_e7'), 'g_in_e7')
        e8 = instance_norm2d(conv2d(leaky_relu(e7), filter_channels * 8, name='g_conv_e8'), 'g_in_e8')

        d1 = deconv2d(tf.nn.relu(e8), filter_channels * 8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm2d(d1, 'g_in_d1'), e7], 3)

        d2 = deconv2d(tf.nn.relu(d1), filter_channels * 8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm2d(d2, 'g_in_d2'), e6], 3)

        d3 = deconv2d(tf.nn.relu(d2), filter_channels * 8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm2d(d3, 'g_in_d3'), e5], 3)

        d4 = deconv2d(tf.nn.relu(d3), filter_channels * 8, name='g_d4')
        d4 = tf.concat([instance_norm2d(d4, 'g_in_d4'), e4], 3)

        d5 = deconv2d(tf.nn.relu(d4), filter_channels * 4, name='g_d5')
        d5 = tf.concat([instance_norm2d(d5, 'g_in_d5'), e3], 3)

        d6 = deconv2d(tf.nn.relu(d5), filter_channels * 2, name='g_d6')
        d6 = tf.concat([instance_norm2d(d6, 'g_in_d6'), e2], 3)

        d7 = deconv2d(tf.nn.relu(d6), filter_channels, name='g_d7')
        d7 = tf.concat([instance_norm2d(d7, 'g_in_d7'), e1], 3)

        d8 = deconv2d(tf.nn.relu(d7), out_channels, stride=1, name='g_d8')

        return tf.nn.tanh(d8), tf.nn.tanh(d7), tf.nn.tanh(d6), tf.nn.tanh(d5), tf.nn.tanh(d4), tf.nn.tanh(
            d3), tf.nn.tanh(d2), tf.nn.tanh(d1)