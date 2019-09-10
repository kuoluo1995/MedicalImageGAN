import tensorflow as tf


def get_metrics_fn_by_name(name):
    return eval(name)


def ssim_metrics(name):
    return 0
