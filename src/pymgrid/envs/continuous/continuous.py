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
        self._nested_action_space = self._get_nested_action_space()
        return flatten_space(self._nested_action_space) if self._flat_spaces else self._nested_action_space

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
            flex_max_prod_marginal_cost = self.modules.flex.get_attrs('max_production', 'marginal_cost')
        except AttributeError:
            flex_production = 0.0
        else:
            zero_marginal_cost_flex = flex_max_prod_marginal_cost['marginal_cost'] == 0
            flex_max_prod = flex_max_prod_marginal_cost.loc[zero_marginal_cost_flex, 'max_production']
            flex_production = flex_max_prod.sum().item()

        return fixed_consumption - flex_production

    def step(self, action):
        return super().step(action, normalized=False)

    def convert_action(self, action, to_microgrid=True):
        # TODO test this. Actions are percentages
        # TODO add a slack module. It should not be in action space and it should be controllable and it
        # should be given an absolute action to balance the rest.

        if to_microgrid:
            relative_action = unflatten(self._nested_action_space, action)
            absolute_action = self.make_absolute(relative_action, self.compute_net_load())

            self._check_action(absolute_action)

            return absolute_action

        relative_action = self.make_relative(action, self.compute_net_load())

        return flatten(self._nested_action_space, relative_action)

    @staticmethod
    def make_relative(action, net_load):
        return NetLoadContinuousMicrogridEnv.convert(action, net_load, 'div')

    @staticmethod
    def make_absolute(action, net_load):
        return NetLoadContinuousMicrogridEnv.convert(action, net_load, 'mul')

    @staticmethod
    def convert(action, net_load, op):
        def _convert(module_act, _op='mul'):
            module_act = module_act.copy().astype(float)

            if op == 'mul':
                module_act[-1] *= net_load
            elif net_load == 0.0 or np.isclose(net_load, 0.0):
                # Same is true when multiplying, but there it's done for us (multiplying by zero is n.p.)
                module_act[-1] = 0.0
            else:
                module_act[-1] /= net_load

            return module_act

        return {name: [_convert(act, op) for act in action_list] for name, action_list in action.items()}

    def _check_action(self, absolute_action):
        if self.check_actions:
            try:
                assert absolute_action in self._nested_action_space
            except AssertionError:
                clipped = self.microgrid_action_space.clip(absolute_action, normalized=False)

                if np.isclose(flatten(self._nested_action_space, absolute_action),
                              flatten(self._nested_action_space, absolute_action)).all():
                    _action = clipped
                else:
                    raise
