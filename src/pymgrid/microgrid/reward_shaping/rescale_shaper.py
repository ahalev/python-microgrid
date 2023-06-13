import numpy as np

from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper


class RescaleShaper(BaseRewardShaper):
    """
    Reward is the original reward scaled by some scale factor.

    ``scaled_reward = self.scale_factor * original_reward ``.

    """

    yaml_tag = u"!RescaleShaper"

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def serializable_state_attributes(self):
        return 'scale_factor'

    def __call__(self, original_reward, step_info, cost_info):
        return self.scale_factor * original_reward


class LearnedScaleRescaleShaper(RescaleShaper):
    yaml_tag = u"!LearnedScaleRescaleShaper"

    def __init__(self, module=('unbalanced_energy', 0)):
        super().__init__(scale_factor=1.0)
        self.module = module

        self.max_load = 0.0
        self.max_renewable = 0.0

    def _update_scale(self, step_info, cost_info):
        total_load = self.compute_total_load(step_info)
        total_renewable = self.compute_total_renewable(step_info)

        if total_load > self.max_load:
            self._max_load = total_load

        if total_renewable > self.max_renewable:
            self._max_renewable = total_renewable

        try:
            baseline_cost_info = cost_info[self.module[0]][self.module[1]]
        except (KeyError, IndexError):
            raise NameError(f"Microgrid has no module in position '{self.module}'")

        baseline = (self.max_load - self.max_renewable) * baseline_cost_info['production_marginal_cost']
        self.scale_factor = 1 / baseline if baseline else self.scale_factor

    def serializable_state_attributes(self):
        return 'module', 'max_load', 'max_renewable', 'scale_factor'

    def __call__(self, original_reward, step_info, cost_info):
        self._update_scale(step_info, cost_info)
        assert self.scale_factor > 0
        return super().__call__(original_reward, step_info, cost_info)
