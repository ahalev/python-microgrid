from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper
from pymgrid.utils.running_mean_std import RunningMeanStd


class StandardizationShaper(BaseRewardShaper):
    yaml_tag = u'StandardizationShaper'

    def __init__(self):
        self.reward_running_mean_std = RunningMeanStd()

    def __call__(self, original_reward, step_info, cost_info):
        self.reward_running_mean_std.update(original_reward)
        return (original_reward - self.reward_running_mean_std.mean) / self.reward_running_mean_std.var
