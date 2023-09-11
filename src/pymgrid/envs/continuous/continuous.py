import numpy as np
import operator

from gym.spaces import Box, Dict, Tuple, flatten_space

from pymgrid.envs.base import BaseMicrogridEnv
from pymgrid.utils.space import MicrogridSpace, extract_builtins, flatten, unflatten


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

    def __init__(self,
                 modules,
                 slack_module=None,
                 clip_actions=True,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 reward_shaping_func=None,
                 trajectory_func=None,
                 flat_spaces=True,
                 observation_keys=None,
                 step_callback=None,
                 reset_callback=None
                 ):

        self._slack_module = slack_module
        self._slack_module_ref = None
        self.clip_actions = clip_actions

        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost,
                         reward_shaping_func=reward_shaping_func,
                         trajectory_func=trajectory_func,
                         flat_spaces=flat_spaces,
                         observation_keys=observation_keys,
                         step_callback=step_callback,
                         reset_callback=reset_callback)

    def _get_action_space(self, remove_redundant_actions=False):
        self._set_slack_module()
        self._nested_action_space = self._get_nested_action_space()
        return flatten_space(self._nested_action_space) if self._flat_spaces else self._nested_action_space

    def _get_nested_action_space(self):

        def extract_box(module_space):
            return Box(low=0.0, high=1.0, shape=module_space.normalized.shape)

        controllable_as = self._modules.controllable.get_attrs('action_space', 'module_type')

        if self._slack_module is not None:
            controllable_as = controllable_as.drop(index=self._slack_module)

        controllable_as['action_space'] = controllable_as['action_space'].apply(extract_box)

        as_builtins = extract_builtins(controllable_as, 'act')
        return as_builtins

    def _set_slack_module(self):
        if self._slack_module is None:
            return

        controllable_modules_dict = self.modules.controllable.to_dict(orient='records')

        msg = ''

        try:
            self._slack_module_ref = controllable_modules_dict[self._slack_module]
        except KeyError:
            controllable_modules_dict_lists = self.modules.controllable.to_dict()
            try:
                self._slack_module_ref = controllable_modules_dict_lists[self._slack_module].item()
                self._slack_module = self._slack_module_ref.name
            except ValueError:
                msg = f"Module name {self._slack_module} does not point to unique controllable candidate."
            except KeyError:
                msg = f"No module '{self._slack_module}' amongst controllable candidates " \
                      f"{list(controllable_modules_dict.keys())}"

            if msg:
                raise NameError(msg)

    def step(self, action):
        return super().step(action, normalized=False)

    def convert_action(self, action, to_microgrid=True, normalize=False):
        if to_microgrid:
            relative_action = unflatten(self._nested_action_space, action)

            self._log_action(relative_action, normalized=False, log_column='net_load_action')

            absolute_action = self.make_absolute(relative_action, self._current_net_load)
            absolute_action = self.clip_action(absolute_action)
            absolute_action = self.add_slack(absolute_action, self._current_net_load)
            self._check_action(absolute_action)
            return absolute_action

        self._check_action(action)
        relative_action = self.make_relative(action, self._current_net_load)
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

    def clip_action(self, action):
        if not self.clip_actions:
            return action

        for module_name, module_list in action.items():
            for module_num, act in enumerate(module_list):
                action[module_name][module_num] = clip_module_action(act, self.modules[(module_name, module_num)])

        return action

    def add_slack(self, action, net_load):
        if self._slack_module is None:
            return action

        slack_action = np.ones(self._slack_module_ref.action_space.shape)
        current_prod = sum([act[-1] for act_list in action.values() for act in act_list])
        remaining_net_load = net_load - current_prod
        slack_action[-1] = remaining_net_load

        module_name, module_num = self._slack_module

        try:
            action[module_name].insert(module_num, slack_action)
        except KeyError:
            assert module_num == 0
            action[module_name] = [slack_action]

        return action

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

    @property
    def slack_module(self):
        return self._slack_module

    @slack_module.setter
    def slack_module(self, value):
        self._slack_module = value
        self._set_slack_module()

    @property
    def slack_module_ref(self):
        return self._slack_module_ref


def clip_module_action(action, module):
    low = -1 * module.max_consumption
    high = module.max_production

    return module.action_space.clip(action, low=low, high=high)
