import numpy as np

from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper


class BaselineShaper(BaseRewardShaper):
    """
    Reward is the original reward minus the baseline of using one module to satisfy net load.

    Parameters
    ----------
    module : tuple
        Module name of the form ``(<module_name>, <module_num>)`` from which to compute the baseline cost.

    relative_to : bool or tuple, default False
        Module to scale rewards to. If True, uses ``module``. Otherwise, expects the name of a module
        in the form ``(<module_name>, <module_num>)`` form which to compute a scale factor.

    """

    yaml_tag = u"!BaselineShaper"

    def __init__(self, module=('grid', 0), relative_to=False):
        self.module = module
        self.baseline_module = self.module if relative_to is True else relative_to

    def compute_baseline_cost(self, step_info, cost_info, baseline_module=None):
        baseline_module = baseline_module or self.module
        net_load = self.compute_net_load(step_info)

        try:
            baseline_cost_info = cost_info[baseline_module[0]][baseline_module[1]]
        except (KeyError, IndexError):
            raise NameError(f"Microgrid has no module in position '{baseline_module}'")

        if net_load > 0:
            baseline_cost = net_load * baseline_cost_info['production_marginal_cost']
        else:
            baseline_cost = -1 * net_load * baseline_cost_info['absorption_marginal_cost']

        return baseline_cost

    def serializable_state_attributes(self):
        return 'module', 'baseline_module'

    def __call__(self, original_reward, step_info, cost_info):
        # Baseline cost is positive, original reward is negative. Equivalent to -1 * (original_cost - baseline_cost).
        baseline_cost = self.compute_baseline_cost(step_info, cost_info)
        reward = original_reward + baseline_cost

        if self.baseline_module:
            return reward / self.compute_baseline_cost(step_info, cost_info, baseline_module=self.baseline_module)

        return reward
