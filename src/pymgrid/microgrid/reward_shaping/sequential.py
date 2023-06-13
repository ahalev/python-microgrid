import functools

from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper


class SequentialShaper(BaseRewardShaper):
    """
    A shaper that is a composition of reward shapers.

    Reward is passed sequentially through passed shapers:

    >>> reward = original_reward
    >>> for shaper in self.shapers:
    >>>     reward = shaper(reward, step_info, cost_info)


    """
    yaml_tag = u"!SequentialShaper"

    def __init__(self, shapers):
        self.shapers = shapers

    def serializable_state_attributes(self):
        return 'shapers',

    def __call__(self, original_reward, step_info, cost_info):
        reward = functools.reduce(lambda res, f: f(res, step_info, cost_info), self.shapers, original_reward)
        return reward
