import inspect
import yaml


class BatteryTransitionModel(yaml.YAMLObject):
    """
    A simple battery transition model.

    In this model, the amount of energy retained is given by ``efficiency``.

    For example, if a microgrid requests 100 kWh of energy and ``efficiency=0.5``, the battery must use
    200 kWh of energy. Alternatively, if a microgrid sends a battery 100 kWh of energy and ``efficiency=0.5``,
    the battery's charge will increase by 50 kWh.

    Parameters
    ----------
    external_energy_change : float
        Amount of energy that is being requested externally.
        If ``energy > 0``, it is energy that is absorbed by the battery -- a charge.
        If ``energy < 0``, it is energy provided by the battery: a discharge.

    min_capacity : float
        Minimum energy that must be contained in the battery.

    max_capacity : float
        Maximum energy that can be contained in the battery.
        If ``soc=1``, capacity is at this maximum.

    max_charge : float
        Maximum amount the battery can be charged in one step.

    max_discharge : float
        Maximum amount the battery can be discharged in one step.

    efficiency : float
        Efficiency of the battery.

    battery_cost_cycle : float
        Marginal cost of charging and discharging.

    state_dict : dict
        State dictionary, with state of charge and current capacity information.

    Returns
    -------
    internal_energy : float
        Amount of energy that the battery must use or will retain given the external amount of energy.

    """

    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader
    yaml_tag = u"!BatteryTransitionModel"

    def __call__(self,
                 external_energy_change,
                 min_capacity,
                 max_capacity,
                 max_charge,
                 max_discharge,
                 efficiency,
                 battery_cost_cycle,
                 state_dict):
        return self.transition(external_energy_change, efficiency)

    def transition(self, external_energy_change, efficiency):
        if external_energy_change < 0:
            return external_energy_change / efficiency
        else:
            return external_energy_change * efficiency

    def __repr__(self):
        params = inspect.signature(self.__init__).parameters
        params = {k: v for k, v in params.items() if k not in ('args', 'kwargs')}
        formatted_params = ', '.join([f'{p}={getattr(self, p)}' for p in params])
        return f'{self.__class__.__name__}({formatted_params})'

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return repr(self) == repr(other)
