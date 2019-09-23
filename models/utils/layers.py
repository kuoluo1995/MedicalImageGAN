import tensorflow as tf
import tensorflow.contrib.slim as slim


def leaky_relu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def conv2d(input, num_outputs, kernel_size=4, stride=2, stddev=0.02, padding='SAME', name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input, num_outputs, kernel_size, stride, padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=None)


def conv3d(input, num_outputs, kernel_size=4, stride=2, stddev=0.02, padding='SAME', name='conv3d'):
    with tf.variable_scope(name):
        return slim.conv3d(input, num_outputs, kernel_size, stride, padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=None)


def deconv2d(input, num_outputs, kernel_size=4, stride=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input, num_outputs, kernel_size, stride, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)


def deconv3d(input, num_outputs, kernel_size=4, stride=2, stddev=0.02, name='deconv3d'):
    with tf.variable_scope(name):
        return slim.conv3d_transpose(input, num_outputs, kernel_size, stride, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)


def instance_norm(input, name='instance_norm'):
    with tf.variable_scope(name):
        depth = input.get_shape()[4]
        scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2, 3], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset
