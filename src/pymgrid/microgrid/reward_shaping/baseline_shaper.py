import numpy as np

from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper


class BaselineShaper(BaseRewardShaper):
    """
    Reward is the original reward scaled by some scale factor.

    ``scaled_reward = self.scale_factor * original_reward ``.

    """

    yaml_tag = u"!BaselineShaper"

    def __init__(self, module=('grid', 0)):
        self.module = module

    @staticmethod
    def compute_net_load(step_info):
        try:
            load_info = step_info['load']
        except KeyError:
            raise NameError("Microgrid has no module with name 'load'")

        try:
            renewable_info = step_info.get('renewable', step_info['pv'])
        except KeyError:
            raise NameError("Microgrid has no module with name 'renewable' or 'pv'.")

        total_load = sum(d['absorbed_energy'] for d in load_info)
        total_renewable = sum(d['provided_energy'] for d in renewable_info)
        return total_load - total_renewable

    def compute_baseline_cost(self, step_info, cost_info):
        net_load = self.compute_net_load(step_info)

        try:
            baseline_cost_info = cost_info[self.module[0]][self.module[1]]
        except (KeyError, IndexError):
            raise NameError(f"Microgrid has no module in position '{self.module}'")

        baseline_cost = net_load * baseline_cost_info['production_marginal_cost']
        return baseline_cost

    def __call__(self, original_reward, step_info, cost_info):
        baseline_cost = self.compute_baseline_cost(step_info, cost_info)
        return original_reward + baseline_cost

