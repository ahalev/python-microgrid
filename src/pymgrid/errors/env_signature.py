from pymgrid import Microgrid, NonModularMicrogrid
from pymgrid.errors import DeprecatedError


def environment_signature_error(cls, modules):
    if isinstance(modules, (Microgrid, NonModularMicrogrid)):
        msg = f'Initializing a {cls} with a microgrid is deprecated as of version 1.5.0. ' \
              f'Use {cls}.from_microgrid() as a drop in replacement.'
    else:
        msg = f'Initializing a {cls} with a scenario integer is deprecated as of version 1.5.0. ' \
              f'Use {cls}.from_scenario() as a drop in replacement.'

    raise DeprecatedError(msg)
