from gym.spaces import flatten as _flatten, unflatten as _unflatten


def flatten(space, x):
    try:
        return _flatten(space, x)
    except Exception as e:
        raise ValueError('Exception encountered when flattening action. See stack trace for details.') from e


def unflatten(space, x):
    try:
        return _unflatten(space, x)
    except Exception as e:
        raise ValueError('Exception encountered when unflattening action. See stack trace for details.') from e
