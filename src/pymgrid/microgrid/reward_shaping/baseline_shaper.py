import numpy as np

from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper


class BaselineShaper(BaseRewardShaper):
    """
    Reward is the original reward minus the baseline of using one module to satisfy net load.

    Parameters
    ----------
    module : tuple
        Module name of the form ``(<module_name>, <module_num>)`` from which to compute the baseline cost.
    """

    yaml_tag = u"!BaselineShaper"

    def __init__(self, module=('grid', 0)):
        self.module = module

    def compute_baseline_cost(self, step_info, cost_info):
        net_load = self.compute_net_load(step_info)

        try:
            baseline_cost_info = cost_info[self.module[0]][self.module[1]]
        except (KeyError, IndexError):
            raise NameError(f"Microgrid has no module in position '{self.module}'")

        baseline_cost = net_load * baseline_cost_info['production_marginal_cost']
        return baseline_cost

    def __call__(self, original_reward, step_info, cost_info):
        # Baseline cost is positive, original reward is negative. Equivalent to -1 * (original_cost - baseline_cost).
        baseline_cost = self.compute_baseline_cost(step_info, cost_info)
        return original_reward + baseline_cost

