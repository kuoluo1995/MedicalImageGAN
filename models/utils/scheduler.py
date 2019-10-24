from functools import partial


def get_scheduler_fn(**config):
    scheduler_fn = eval(config['name'])
    scheduler_fn = partial(scheduler_fn, **config)
    return scheduler_fn


def linear_decay(epoch, learning_rate, total_epoch, decay_epoch, **kwargs):
    if epoch < decay_epoch:
        lr = learning_rate
    else:
        lr = learning_rate * (1 - 1 / (total_epoch - decay_epoch) * (epoch - decay_epoch))
    return lr
