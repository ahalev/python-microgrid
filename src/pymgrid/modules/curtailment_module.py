import numpy as np

import yaml

from pymgrid.modules.base import BaseMicrogridModule, BaseTimeSeriesMicrogridModule
from pymgrid.modules.module_container import ModuleContainer


class CurtailmentModule(BaseMicrogridModule):
    """
    A curtailment module for renewables.

    The classic examples of renewables are photovoltaics (PV) and wind turbines. This module allows for their
    production to be curtailed without incurring costs (or incurring costs different from other overgeneration).

    Parameters
    ----------
    modules_to_curtail : list-like or None, default None
        List of modules to curtail or None, in which case all curtailment will apply to all fixed source modules,
        of which renewables are by default. Can contain any of the following:

        * :class:`.BaseMicrogridModule` :
           Specific modules to curtail: subclasses of the base module.

        * tuple, length two :
           tuple of the form ``(module_name, module_number)`` pointing to a specific module, e.g. ``('renewable', 0)``.

        * str :
          Name of a particular module type, e.g. ``'renewable'``. All modules in :class:`.Microgrid.modules```['renewable']
          will be included.

        * None :
          Use all fixed source modules, of which renewables are.

    curtailment_cost : float, default 0.0
        Unit cost of curtailment.

    raise_errors : bool, default False
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    """
    module_type = ('curtailment', 'flex')
    yaml_tag = u"!CurtailmentModule"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    def __init__(self,
                 modules_to_curtail=None,
                 initial_step=0,
                 curtailment_cost=0.0,
                 normalized_action_bounds=(0, 1),
                 raise_errors=False):

        super().__init__(raise_errors,
                         initial_step=initial_step,
                         normalized_action_bounds=normalized_action_bounds,
                         provided_energy_name=None,
                         absorbed_energy_name='curtailment')

        self._modules_to_curtail = modules_to_curtail
        self.curtailment_cost = curtailment_cost

        self._curtailment_modules = None
        self._next_max_consumption = None

    def reset(self):
        pass

    def setup(self, module_container):
        """

        Parameters
        ----------
        microgrid : :class:`pymgrid.Microgrid`

        Returns
        -------

        """
        if self._modules_to_curtail is None:
            curtailment_modules = module_container.fixed.source.to_dict()

        else:
            curtailment_modules = []

            for module_ref in self._modules_to_curtail:
                curtailment_modules.extend(self._get_modules_from_ref(module_container, module_ref))

        self._curtailment_modules = ModuleContainer(curtailment_modules, set_names=False)
        self._update_max_consumption()

    def _get_modules_from_ref(self, modules, ref):
        if isinstance(ref, BaseMicrogridModule):  # Module
            referenced_modules = [ref]

        elif isinstance(ref, tuple):  # Name of a module, e.g. ('renewable', 0)
            if ref == self.name:
                raise NameError('Cannot reference itself.')

            try:
                referenced_modules = [modules[ref[0]][ref[1]]]
            except (KeyError, IndexError):
                raise NameError(f'Module {ref} not found.')

        elif isinstance(ref, str):  # Name of a module type, e.g. 'renewable'
            try:
                referenced_modules = modules[ref]
            except KeyError:
                raise NameError(f'Module {ref} not found.')
        else:
            raise TypeError(f"Unrecognized module reference '{ref}'.")

        return referenced_modules

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_sink

        if not self._curtailment_modules:
            raise RuntimeError('Must call RenewableCurtailmentModule.setup before usage!')

        curtailment = min(external_energy_change, self.max_consumption)
        info = {'absorbed_energy': curtailment}
        reward = -1.0 * self.get_cost(curtailment)

        done = self._update_max_consumption()

        return reward, done, info

    def get_cost(self, curtailment):
        return self.curtailment_cost * curtailment

    def _update_max_consumption(self):
        try:
            self._next_max_consumption = self._curtailment_modules.get_attrs('max_production').sum().item()
            return False
        except IndexError:
            assert self._current_step == self._curtailment_modules.get_attrs('final_step', unique=True).item() - 1
            self._next_max_consumption = 0.0
            return True

    def _state_dict(self):
        return dict()

    @property
    def state(self):
        return np.array([])

    @property
    def min_obs(self):
        return np.array([])

    @property
    def max_obs(self):
        return np.array([])

    @property
    def min_act(self):
        # TODO (ahalev) find a better bound
        return -np.inf

    @property
    def max_act(self):
        return 0.0

    @property
    def max_consumption(self):
        module_current_step = self._curtailment_modules.get_attrs('current_step', unique=True).item()
        if not self._current_step == module_current_step - 1:
            raise RuntimeError(f'self.current_step={self._current_step} is not one less than curtailment module current'
                               f'step ({module_current_step}). This module should only be called after curtailment'
                               f'modules.')

        return self._next_max_consumption

    @property
    def is_sink(self):
        return True

    @property
    def absorption_marginal_cost(self):
        return self.curtailment_cost
