import tensorflow as tf
import numpy as np

EPS = 1e-12


def get_int_shape(shape):
    return [-1 if i is None else i for i in shape.as_list()]


min_filters = 256  # 512


def get_out_channels(num):
    return min(1024 // (2 ** (num * 1)), min_filters)


def get_weight(shape, gain):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    wscale = tf.constant(np.float32(std), name='wscale')
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale


def leaky_relu(x, a=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def reflection_padding(x, pad, name='reflection_padding'):
    with tf.variable_scope(name):
        padding = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        return tf.keras.layers.Lambda(lambda b: tf.pad(b, padding, mode='REFLECT'))(x)


def sobel_edges(batch_input, name='sobel_edge'):
    with tf.variable_scope(name):
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
        sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
        filtered_x = tf.nn.conv2d(batch_input, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
        filtered_y = tf.nn.conv2d(batch_input, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
        return filtered_x, filtered_y


def down_sampling(batch_input, i):
    return tf.layers.average_pooling2d(batch_input, 2 ** i, 2 ** i)


def upscale(x, scale):
    _, h, w, _ = get_int_shape(x.get_shape())
    return tf.image.resize_nearest_neighbor(x, (h * scale, w * scale))


def minbatch_concat(x, eps=1e-8, averaging='all'):
    shape = get_int_shape(x.get_shape())
    adjusted_std = lambda _x: tf.sqrt(
        tf.reduce_mean((_x - tf.reduce_mean(_x, axis=0, keep_dims=True)) ** 2, axis=0, keep_dims=True) + eps)
    vals = adjusted_std(x)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keepdims=True)
    vals = tf.tile(vals, multiples=[shape[0], shape[1], shape[2], 1])
    return tf.concat([x, vals], axis=3)

def pg_fully_connect(x, filters, gain, name='my_dense'):
    shape = get_int_shape(x.get_shape())
    with tf.variable_scope(name):
        w = get_weight([shape[1], filters], gain=gain)
        w = tf.cast(w, x.dtype)
        bias = tf.get_variable('bias', [filters], initializer=tf.constant_initializer(0.0))
        output = tf.matmul(x, w) + bias
        return output

def conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same', name='conv2d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        if padding == 'reflect':
            batch_input = reflection_padding(batch_input, kernel_size // 2)
            padding = 'valid'
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                                kernel_initializer=initializer)


def pg_conv2d(x, filters, kernel_size, strides, padding, gain, name='pg_conv2d'):
    shape = get_int_shape(x.get_shape())
    with tf.variable_scope(name):
        w = get_weight([kernel_size, kernel_size, shape[-1], filters], gain=gain)
        w = tf.cast(w, x.dtype)
        if padding == 'OTHER':
            padding = 'VALID'
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'CONSTANT')
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
        biases = tf.get_variable('biases', [filters], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(tf.nn.bias_add(x, biases), get_int_shape(x.get_shape()))
        return x


def res_block2d(batch_input, out_channels, padding='reflect', name='res_block2d'):
    with tf.variable_scope(name):
        output = batch_input
        output = leaky_relu(
            instance_norm2d(conv2d(output, out_channels, kernel_size=3, strides=(1, 1), padding=padding)))
        output = instance_norm2d(conv2d(output, out_channels, kernel_size=3, strides=(1, 1), padding=padding))
        return batch_input + output


def separable_conv2d(batch_input, out_channels, name='separable_conv2d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)


def discrim_conv2d(batch_input, out_channels, stride, name='discrim_conv2d'):
    with tf.variable_scope(name):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding='valid',
                                kernel_initializer=tf.random_normal_initializer(0, 0.02))


def conv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding='same', name='conv3d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        if padding == 'reflect':
            batch_input = reflection_padding(batch_input, kernel_size // 2)
            padding = 'valid'
        return tf.layers.conv3d(batch_input, out_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                                kernel_initializer=initializer)


def deconv2d(batch_input, out_channels, kernel_size=4, name='deconv2d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=kernel_size, strides=(2, 2),
                                          padding='same', kernel_initializer=initializer)


def separable_deconv2d(batch_input, out_channels, name='separable_deconv2d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding='same',
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)


def deconv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), name='deconv3d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv3d_transpose(batch_input, out_channels, kernel_size=kernel_size, strides=strides,
                                          padding='same', kernel_initializer=initializer)


def discrim_conv3d(batch_input, out_channels, strides, name='discrim_conv3d'):
    with tf.variable_scope(name):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        return tf.layers.conv3d(padded_input, out_channels, kernel_size=4, strides=strides,
                                padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.02))


def pix_norm(x, eps=1e-8, name='pixel_norm'):
    with tf.variable_scope(name):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keep_dims=True) + eps)


def instance_norm2d(x, name='instance_norm2d'):
    with tf.variable_scope(name):
        channels = x.shape[-1]
        gamma = tf.get_variable('gamma', [channels],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        beta = tf.get_variable('beta', [channels], initializer=tf.constant_initializer(0.0))

        epsilon = 1e-5
        mean, sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        _x = (x - mean) / (tf.sqrt(sigma + epsilon))

        norm_x = _x * gamma + beta
        return norm_x


def instance_norm3d(x, name='instance_norm3d'):
    with tf.variable_scope(name):
        channels = x.shape[-1]
        gamma = tf.get_variable('gamma', [channels],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        beta = tf.get_variable('beta', [channels], initializer=tf.constant_initializer(0.0))

        epsilon = 1e-5
        mean, sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        _x = (x - mean) / (tf.sqrt(sigma + epsilon))
        norm_x = _x * gamma + beta
        return norm_x


def batch_norm2d(x, name='batch_norm2d'):
    return tf.layers.batch_normalization(x, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02), name=name)
    # with tf.variable_scope(name):
    #     channels = x.shape[-1]
    #     gamma = tf.get_variable('gamma', [channels],
    #                             initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    #     beta = tf.get_variable('beta', [channels], initializer=tf.constant_initializer(0.0))
    #     momentum = tf.get_variable("momentum", [channels], initializer=tf.constant_initializer(1.0),
    #                                constraint=lambda y: tf.clip_by_value(y, clip_value_min=0.0, clip_value_max=1.0))
    #
    #     epsilon = 1e-5
    #     mean, sigma = tf.nn.moments(x, axes=[0, 2, 3], keep_dims=True)
    #     _x = (x - mean) / (tf.sqrt(sigma + epsilon))
    #
    #     norm_x = momentum * _x + (1 - momentum) * _x
    #     norm_x = norm_x * gamma + beta
    #     return norm_x
