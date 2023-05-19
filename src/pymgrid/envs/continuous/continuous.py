import numpy as np
import operator

from gym.spaces import Box, Dict, Tuple, flatten_space, flatten, unflatten

from pymgrid.envs.base import BaseMicrogridEnv
from pymgrid.utils.space import MicrogridSpace, extract_builtins


class ContinuousMicrogridEnv(BaseMicrogridEnv):
    _nested_action_space = None
    check_actions = True

    def _get_nested_action_space(self):
        return Dict({name: Tuple([module.action_space['normalized'] for module in modules_list])
                                 for name, modules_list in self.modules.controllable.iterdict()})

    def _get_action_space(self, remove_redundant_actions=False):
        self._nested_action_space = self._get_nested_action_space()
        return flatten_space(self._nested_action_space) if self._flat_spaces else self._nested_action_space

    def convert_action(self, action, to_microgrid=True, normalize=False):
        if to_microgrid:
            converted = unflatten(self._nested_action_space, action)
            if normalize:
                converted = self.microgrid_action_space.normalize(converted)

            return converted

        if normalize:
            _action = self.microgrid_action_space.normalize(action)
        else:
            _action = action

        if self.check_actions:
            try:
                assert _action in self._nested_action_space
            except AssertionError:
                clipped = self.microgrid_action_space.clip(_action, normalized=normalize)

                if np.isclose(flatten(self._nested_action_space, clipped),
                              flatten(self._nested_action_space, _action)).all():
                    _action = clipped
                else:
                    raise

        return flatten(self._nested_action_space, _action)


class NetLoadContinuousMicrogridEnv(BaseMicrogridEnv):
    """
    A continuous action space environment where actions are percentage of the net load to be fulfilled by each module

    E.g. action = [a_battery, a_grid]:
    a_battery is % of net load to be fulfilled by battery,
    a_grid is % to be fulfilled by grid
        """
    _nested_action_space = None
    check_actions = True

    def _get_action_space(self, remove_redundant_actions=False):
        self._nested_action_space = self._get_nested_action_space
        return flatten_space(self._nested_action_space) if self._flat_spaces else self._nested_action_space

    @property
    def _get_nested_action_space(self):
        as_builtins_unnormalized = extract_builtins(self._modules.get_attrs('action_space', 'module_type', as_pandas=False), 'act')
        as_builtins_normalized = Dict({
            name: Tuple([Box(low=0, high=1, shape=action_space.shape) for action_space in as_list])
            for name, as_list in as_builtins_unnormalized.items()
        })

        microgrid_space = MicrogridSpace(as_builtins_unnormalized, as_builtins_normalized)

        return Dict({name: Tuple([Box(low=0, high=1, shape=module.action_space.shape) for module in modules_list])
                     for name, modules_list in self.modules.controllable.iterdict()})

    def compute_net_load(self):
        """
        Compute the net load at the current step.

        Net load is load minus renewables.
        -------

        Returns
        -------
        net_load : float
            Net load.

        """
        try:
            fixed_consumption = self.modules.fixed.get_attrs('max_consumption').sum().item()
        except AttributeError:
            fixed_consumption = 0.0

        try:
            flex_production = self.modules.flex.get_attrs('max_production').sum().item()
        except AttributeError:
            flex_production = 0.0

        return fixed_consumption - flex_production

    def convert_action(self, action, to_microgrid=True):
        # TODO test this. Actions are percentages
        # TODO add a slack module. It should not be in action space and it should be controllable and it
        # should be given an absolute action to balance the rest.

        if to_microgrid:
            relative_action = unflatten(self._nested_action_space, action)
            absolute_action = self._get_absolute_action(relative_action)

            self._check_action(absolute_action)

            return absolute_action

        relative_action = self._get_relative_action(action)

        return flatten(self._nested_action_space, relative_action)

    def _get_absolute_action(self, relative_action):
        # TODO test this
        net_load = self.compute_net_load()

        return MicrogridSpace.dict_op(relative_action, net_load, operator.mul)

    def _get_relative_action(self, absolute_action):
        # TODO test this
        net_load = self.compute_net_load()

        return MicrogridSpace.dict_op(absolute_action, net_load, operator.truediv)

    def _check_action(self, absolute_action):
        if self.check_actions:
            try:
                assert absolute_action in self._nested_action_space
            except AssertionError:
                clipped = self.microgrid_action_space.clip(absolute_action)

                if np.isclose(flatten(self._nested_action_space, absolute_action),
                              flatten(self._nested_action_space, absolute_action)).all():
                    _action = clipped
                else:
                    raise
