from functools import partial


def get_scheduler_fn(config):
    scheduler_fn = eval(config['name'])
    scheduler_fn = partial(scheduler_fn, **config)
    return scheduler_fn


def linear_policy(epoch):
    return 1e-3
