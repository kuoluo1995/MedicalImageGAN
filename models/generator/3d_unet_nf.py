import tensorflow as tf
from models.utils.layers import conv3d, instance_norm3d, leaky_relu, deconv3d


def build_model(image, out_channels, filter_channels=64, reuse=False, name='3dunet', is_training=False, **kwargs):
    dropout_rate = 0.5 if is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        e1 = instance_norm3d(conv3d(image, filter_channels, stride=1, name='g_conv_e1'), 'g_in_e1')
        e2 = instance_norm3d(conv3d(leaky_relu(e1), filter_channels * 2, name='g_conv_e2'), 'g_in_e2')
        e3 = instance_norm3d(conv3d(leaky_relu(e2), filter_channels * 4, name='g_conv_e3'), 'g_in_e3')
        e4 = instance_norm3d(conv3d(leaky_relu(e3), filter_channels * 8, stride=(2, 2, 1), name='g_conv_e4'), 'g_in_e4')
        e5 = instance_norm3d(conv3d(leaky_relu(e4), filter_channels * 8, stride=(2, 2, 1), name='g_conv_e5'), 'g_in_e5')
        e6 = instance_norm3d(conv3d(leaky_relu(e5), filter_channels * 8, stride=(2, 2, 1), name='g_conv_e6'), 'g_in_e6')
        e7 = instance_norm3d(conv3d(leaky_relu(e6), filter_channels * 8, stride=(2, 2, 1), name='g_conv_e7'), 'g_in_e7')

        d1 = deconv3d(tf.nn.relu(e7), filter_channels * 8, stride=(2, 2, 1), name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm3d(d1, 'g_in_d1'), e6], 4)

        d2 = deconv3d(tf.nn.relu(d1), filter_channels * 8, stride=(2, 2, 1), name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm3d(d2, 'g_in_d2'), e5], 4)

        d3 = deconv3d(tf.nn.relu(d2), filter_channels * 8, stride=(2, 2, 1), name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm3d(d3, 'g_in_d3'), e4], 4)

        d4 = deconv3d(tf.nn.relu(d3), filter_channels * 4, stride=(2, 2, 1), name='g_d4')
        d4 = tf.concat([instance_norm3d(d4, 'g_in_d4'), e3], 4)

        d5 = deconv3d(tf.nn.relu(d4), filter_channels * 2, name='g_d5')
        d5 = tf.concat([instance_norm3d(d5, 'g_in_d5'), e2], 4)

        d6 = deconv3d(tf.nn.relu(d5), filter_channels, name='g_d6')
        d6 = tf.concat([instance_norm3d(d6, 'g_in_d6'), e1], 4)

        d7 = deconv3d(tf.nn.relu(d6), out_channels, stride=1, name='g_d7')

        return tf.nn.tanh(d7)
