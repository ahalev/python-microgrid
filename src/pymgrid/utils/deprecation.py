import functools
import warnings


def deprecation_warning(successor, msg=None):
    # TODO (ahalev) use obj_linkcode here
    """
    Decorator for warning of future deprecation

    Raises a DeprecationWarning on the wrapped function and suggests using `successor` instead.
    If msg is not None, raises a future warning with said message. In this case `successor` is ignored.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            _msg = msg or f"Function '{func.__qualname__}' is deprecated and will be removed in a future version. "\
                          f"Use '{successor}' instead."

            warnings.warn(_msg, category=DeprecationWarning)

            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def deprecation_err(successor, msg=None):
    # TODO (ahalev) use obj_linkcode here
    """
    Decorator for warning of future deprecation

    Raises a DeprecationWarning on the wrapped function and suggests using `successor` instead.
    If msg is not None, raises a future warning with said message. In this case `successor` is ignored.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # TODO (ahalev) get this version
            _msg = msg or f"Function '{func.__name__}' is deprecated as of version {'version'}. "\
                          f"Use '{successor}' instead."

            raise DeprecatedError(_msg)

        return wrapper
    return decorator


class DeprecatedError(Exception):
    pass
