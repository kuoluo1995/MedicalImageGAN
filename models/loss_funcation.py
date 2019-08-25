import tensorflow as tf


def get_loss_fn_by_name(name):
    return eval(name)


def lsgan_loss(logits, labels):
    return tf.reduce_mean((logits - labels) ** 2)


def sce_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def abs_loss(logits, labels):
    return tf.reduce_mean(tf.abs(logits - labels))
