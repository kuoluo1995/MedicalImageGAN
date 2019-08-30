import tensorflow as tf
from models.utils.layers import conv2d, instance_norm, deconv2d


def residule_block(x, num_outputs, kernel_size=3, stride=1, name='res'):
    padding = int((kernel_size - 1) / 2)
    y = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
    y = instance_norm(conv2d(y, num_outputs, kernel_size, stride, padding='VADLID', name=name + '_c1'), name + '_bn1')
    y = tf.nn.relu(y)
    y = tf.pad(y, [[0, 0], [padding, padding], [padding, padding], [0, 0]], 'REFLECT')
    y = instance_norm(conv2d(y, num_outputs, kernel_size, stride, padding='VALID', name=name + '_c2'), name + '_bn2')
    return y + x


def build_model(image, out_channels, filter_channels=64, reuse=False, name='resnet', **kwargs):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        c1 = tf.nn.relu(instance_norm(conv2d(c0, filter_channels, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, filter_channels * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, filter_channels * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, filter_channels * 4, name='g_r1')
        r2 = residule_block(r1, filter_channels * 4, name='g_r2')
        r3 = residule_block(r2, filter_channels * 4, name='g_r3')
        r4 = residule_block(r3, filter_channels * 4, name='g_r4')
        r5 = residule_block(r4, filter_channels * 4, name='g_r5')
        r6 = residule_block(r5, filter_channels * 4, name='g_r6')
        r7 = residule_block(r6, filter_channels * 4, name='g_r7')
        r8 = residule_block(r7, filter_channels * 4, name='g_r8')
        r9 = residule_block(r8, filter_channels * 4, name='g_r9')

        d1 = tf.nn.relu(instance_norm(deconv2d(r9, filter_channels * 2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))
        d2 = tf.nn.relu(instance_norm(deconv2d(d1, filter_channels, 3, 2, name='g_d2_dc'), 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        d2 = conv2d(d2, out_channels, 7, 1, padding='VALID', name='g_pred_c')

        pred = tf.nn.tanh(d2)
        return pred
