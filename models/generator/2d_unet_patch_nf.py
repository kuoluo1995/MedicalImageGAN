import tensorflow as tf
from models.utils.layers import conv2d, instance_norm2d, leaky_relu, deconv2d


def build_model(batch_input, out_channels, filter_channels=64, reuse=False, name='unet', is_training=False, **kwargs):
    dropout_rate = 0.5 if is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        e1 = instance_norm2d(conv2d(batch_input, filter_channels, name='g_conv_e1'), 'g_in_e1')
        e2 = instance_norm2d(conv2d(leaky_relu(e1), filter_channels * 2, name='g_conv_e2'), 'g_in_e2')
        e3 = instance_norm2d(conv2d(leaky_relu(e2), filter_channels * 4, name='g_conv_e3'), 'g_in_e3')
        e4 = instance_norm2d(conv2d(leaky_relu(e3), filter_channels * 8, name='g_conv_e4'), 'g_in_e4')
        e5 = instance_norm2d(conv2d(leaky_relu(e4), filter_channels * 8, name='g_conv_e5'), 'g_in_e5')
        e6 = instance_norm2d(conv2d(leaky_relu(e5), filter_channels * 8, name='g_conv_e6'), 'g_in_e6')
        e7 = instance_norm2d(conv2d(leaky_relu(e6), filter_channels * 8, name='g_conv_e7'), 'g_in_e7')
        e8 = instance_norm2d(conv2d(leaky_relu(e7), filter_channels * 8, name='g_conv_e8'), 'g_in_e8')

        d8 = instance_norm2d(deconv2d(tf.nn.relu(e8), filter_channels * 8, name='g_conv_d8'), 'g_in_d8')
        d8 = tf.nn.dropout(d8, keep_prob=dropout_rate)
        d8 = tf.concat([d8, e7], 3)

        d7 = instance_norm2d(deconv2d(tf.nn.relu(d8), filter_channels * 8, name='g_conv_d7'), 'g_in_d7')
        d7 = tf.nn.dropout(d7, keep_prob=dropout_rate)
        d7 = tf.concat([d7, e6], 3)

        d6 = instance_norm2d(deconv2d(tf.nn.relu(d7), filter_channels * 8, name='g_conv_d6'), 'g_in_d6')
        d6 = tf.nn.dropout(d6, dropout_rate)
        d6 = tf.concat([d6, e5], 3)

        d5 = instance_norm2d(deconv2d(tf.nn.relu(d6), filter_channels * 8, name='g_conv_d5'), 'g_in_d5')
        d5 = tf.concat([d5, e4], 3)

        d4 = instance_norm2d(deconv2d(tf.nn.relu(d5), filter_channels * 4, name='g_conv_d4'), 'g_in_d4')
        d4 = tf.concat([d4, e3], 3)

        d3 = instance_norm2d(deconv2d(tf.nn.relu(d4), filter_channels * 2, name='g_conv_d3'), 'g_in_d3')
        d3 = tf.concat([d3, e2], 3)

        d2 = instance_norm2d(deconv2d(tf.nn.relu(d3), filter_channels, name='g_conv_d2'), 'g_in_d2')
        d2 = tf.concat([d2, e1], 3)

        d1 = deconv2d(tf.nn.relu(d2), out_channels, name='g_conv_d1')
        return tf.nn.tanh(d1)
