from functools import partial


def get_scheduler_fn(**config):
    scheduler_fn = eval(config['name'])
    scheduler_fn = partial(scheduler_fn, **config)
    return scheduler_fn


def linear_decay(epoch, learning_rate, total_epoch, epoch_decay, **kwargs):
    if epoch < epoch_decay:
        lr = learning_rate
    else:
        lr = learning_rate * (1 - 1 / (total_epoch - epoch_decay) * (epoch - epoch_decay))
    return lr
