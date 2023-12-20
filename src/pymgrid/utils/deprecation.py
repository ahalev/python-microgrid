import functools
import warnings

from pymgrid.errors import PymgridDeprecationWarning, DeprecatedError


def deprecation_warning(successor, msg=None):
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

            warnings.warn(_msg, category=PymgridDeprecationWarning)

            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def deprecation_err(successor, version=None, msg=None):
    """
    Decorator for raising error on deprecated method or function.

    Raises a DeprecatedError on the wrapped function and suggests using `successor` instead.
    If msg is not None, raises a future warning with said message. In this case `successor` is ignored.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            version_msg = f' as of version {version}' if version else ''
            _msg = msg or f"Function '{func.__name__}' is deprecated{version_msg}. "\
                          f"Use '{successor}' instead."

            raise DeprecatedError(_msg)

        return wrapper
    return decorator
