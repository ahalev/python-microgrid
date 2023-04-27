import yaml

from contextlib import contextmanager


@contextmanager
def dry_run(pymgrid_object):
    """
    A context manager to test modifications of a pymgrid object without modifying said object.

    Parameters
    ----------
    pymgrid_object : :class:`pymgrid.Microgrid` or :class:`pymgrid.modules.base.BaseMicrogridModule`
        Microgrid or module to try run

    Returns
    -------
    test_object : copy of object to test usage of, without modifying original object.

    Examples
    --------
    >>> from pymgrid.modules import BatteryModule
    >>> module = BatteryModule(0, 100, 50, 50, 0.9, init_soc=0.5)
    >>> with dry_run(module) as test_module:
    >>>     test_module.step(test_module.max_act, normalized=False)
    >>>     print(f'Current step: {test_module.current_step}; current charge: {test_module.current_charge}')
    Current step: 1; current charge: 0.0
    >>> print(f'Current step: {module.current_step}; current charge: {module.current_charge}')
    Current step: 0; current charge: 50.0

    """

    serialized = yaml.safe_dump(pymgrid_object)

    try:
        yield pymgrid_object
    finally:
        deserialized = yaml.safe_load(serialized)
        try:
            # Module
            data = deserialized._serialize_state_attributes()
        except AttributeError:
            # Microgrid
            data = deserialized._serialization_data()
            pymgrid_object._modules = deserialized.modules

        pymgrid_object.deserialize(data)