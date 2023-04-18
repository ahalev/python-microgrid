from gym.spaces import Dict, Tuple, flatten_space, unflatten

from pymgrid.envs.base import BaseMicrogridEnv


class ContinuousMicrogridEnv(BaseMicrogridEnv):
    _nested_action_space = None

    def _get_nested_action_space(self):
        return Dict({name: Tuple([module.action_space['normalized'] for module in modules_list])
                                 for name, modules_list in self.modules.controllable.iterdict()})

    def _get_action_space(self, remove_redundant_actions=False):
        self._nested_action_space = self._get_nested_action_space()
        return flatten_space(self._nested_action_space) if self._flat_spaces else self._nested_action_space

    def convert_action(self, action):
        return unflatten(self._nested_action_space, action)

    def run(self, action, normalized=True):
        warn('run() should not be called directly in environments.')
        return super().run(action, normalized=normalized)
