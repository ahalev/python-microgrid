import inspect
import os
import pymgrid


def obj_linkcode(obj):
    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno, source = None, ''
    except OSError:
        lineno, source = None, ''

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(pymgrid.__file__))

    return f'https://github.com/ahalev/python-microgrid/tree/v{pymgrid.__version__}/src/pymgrid/{fn}{linespec}'
