import tensorflow as tf

EPS = 1e-12


def leaky_relu(x, a=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def conv2d(batch_input, out_channels, name='conv2d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same',
                                kernel_initializer=initializer)


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


def conv3d(batch_input, out_channels, name='conv3d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding='same',
                                kernel_initializer=initializer)


def deconv2d(batch_input, out_channels, name='deconv2d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding='same',
                                          kernel_initializer=initializer)


def separable_deconv2d(batch_input, out_channels, name='separable_deconv2d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding='same',
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)


def deconv3d(batch_input, out_channels, name='deconv3d'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding='same',
                                kernel_initializer=initializer)


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
