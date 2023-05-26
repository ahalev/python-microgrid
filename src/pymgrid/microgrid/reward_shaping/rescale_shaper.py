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

    def __call__(self, original_reward, step_info, cost_info):
        return self.scale_factor * original_reward
