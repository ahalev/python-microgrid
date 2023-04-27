import numpy as np

from gym.spaces import Dict, Tuple, flatten_space, flatten, unflatten

from pymgrid.envs.base import BaseMicrogridEnv


class ContinuousMicrogridEnv(BaseMicrogridEnv):
    _nested_action_space = None

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

        try:
            assert _action in self._nested_action_space
        except AssertionError:
            clipped = self.microgrid_action_space.clip(_action, normalized=normalize)
            if np.isclose(clipped, _action):
                _action = clipped
            else:
                raise

        return flatten(self._nested_action_space, _action)

