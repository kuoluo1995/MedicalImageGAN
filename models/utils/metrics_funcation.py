import tensorflow as tf
from skimage.measure import compare_ssim, compare_mse, compare_psnr


def get_metrics_fn_by_name(name):
    return eval(name)


def ssim_metrics(synthesis_image, original_image):
    # ssim = compare_ssim(synthesis_image, original_image)
    ssim = tf.image.ssim(synthesis_image, original_image, max_val=1.0)
    return tf.reduce_mean(ssim)


def mse_metrics(synthesis_image, original_image):
    mse = compare_mse(synthesis_image, original_image)
    return mse


def psnr_metrics(synthesis_image, original_image):
    psnr = compare_psnr(synthesis_image, original_image)
    return psnr
