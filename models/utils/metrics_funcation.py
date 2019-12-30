import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops


def get_metrics_fn_by_name(name):
    return eval(name)


def ssim_metrics(img1, img2):
    # ssim = tf.image.ssim(img1, img2, max_val=2.0)
    # tf.reduce_mean(ssim)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1_mean = img1.mean()
    img2_mean = img2.mean()
    img1_sigma = np.sqrt(((img1 - img1_mean) ** 2).mean())
    img2_sigma = np.sqrt(((img2 - img2_mean) ** 2).mean())
    img12_sigma = ((img1 - img1_mean) * (img2 - img2_mean)).mean()
    k1, k2, L = 0.01, 0.03, 2.0
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * img1_mean * img2_mean + C1) / (img1_mean ** 2 + img2_mean ** 2 + C1)
    c12 = (2 * img1_sigma * img2_sigma + C2) / (img1_sigma ** 2 + img2_sigma ** 2 + C2)
    s12 = (img12_sigma + C3) / (img1_sigma * img2_sigma + C3)
    ssim = l12 * c12 * s12
    return ssim


def mse_metrics(img1, img2):
    # mse = math_ops.reduce_mean(math_ops.squared_difference(img1, img2))
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return mse


def nmse_metrics(img1, img2):
    # mse = math_ops.reduce_mean(math_ops.squared_difference(img1, img2))
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1_mean = img1.mean()
    img2_mean = img2.mean()
    mse = np.mean((img1 - img1_mean + img2 - img2_mean) ** 2)
    return mse


def psnr_metrics(img1, img2):
    # psnr = tf.image.ssim(img1, img2, max_val=1.0)
    # tf.reduce_mean(psnr)
    img1 = img1.astype(np.float64) * 2.0
    img2 = img2.astype(np.float64) * 2.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 * 1.0 / mse)


def dice_coefficient(logits, labels, axis=(1, 2, 3), smooth=1e-5):
    intersection = tf.reduce_sum(logits * labels, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth

    denominator = tf.reduce_sum(logits * logits, axis=axis) + tf.reduce_sum(labels * labels, axis=axis) + smooth
    coefficient = numerator / denominator
    return tf.reduce_mean(coefficient)
