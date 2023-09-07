import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Union

from pymgrid import Microgrid
from pymgrid.envs import NetLoadContinuousMicrogridEnv


def analyze_reward_shaping_func(microgrid: Microgrid,
                                reward_shaping_func=lambda x: x,
                                policy: Union[callable, str]='random',
                                n_steps=1000,
                                plot=False):

    if policy == 'random':
        policy = get_random_policy(microgrid)

    microgrid.reward_shaping_func = reward_shaping_func

    out = []

    done = False
    obs = microgrid.reset()

    for _ in tqdm(range(n_steps)):
        try:
            obs, reward, done, info = microgrid.step(policy(obs))
        except AttributeError:
            obs, reward, done, info = microgrid.run(policy(obs))

        if done:
            out.append(microgrid.log)
            obs = microgrid.reset()

        # TODO put the cumsum of rewards and shaped rewards here -- that's we want to compare

    out.append(microgrid.log)

    df = pd.concat(out)

    if plot:
        plot_results = df.loc[:, pd.IndexSlice['balance', 0]]
        plot_results = plot_results[['reward', 'shaped_reward']]

        import_price = df[('grid', 0, 'import_price_current')]
        import_price.name = 'import_price'
        non_loss_load_reward = df[('balance', 0, 'reward')] - df[('unbalanced_energy', 0, 'reward')]
        non_loss_load_reward.name = 'non_loss_load_reward'

        plot_results = pd.concat([plot_results, import_price, non_loss_load_reward], axis=1)
        sns.scatterplot(data=plot_results, x='reward', y='shaped_reward', hue='non_loss_load_reward', marker='.')
        plt.show()

    return df


def get_random_policy(microgrid):
    try:
        action_space = microgrid.action_space
    except AttributeError:
        action_space = microgrid.microgrid_action_space

    def policy(obs):
        return np.array([1.0])
        # return action_space.sample()

    return policy


class ActionGenerator:
    def __init__(self, microgrid, n_actions=int(1e6)):
        try:
            self.action_space = microgrid.action_space
        except AttributeError:
            self.action_space = microgrid.microgrid_action_space

        self.actions = [self.action_space.sample() for _ in range(n_actions)]
        self._counter = 0

    def reset(self):
        self._counter = 0
        return self.get_action

    def get_action(self, obs):
        self._counter += 1
        try:
            return self.actions[self._counter-1]
        except IndexError:
            raise IndexError('All actions exhausted!')


if __name__ == '__main__':
    from pymgrid.microgrid.reward_shaping import BaselineShaper, SequentialShaper, RescaleShaper, LearnedScaleRescaleShaper

    microgrid = NetLoadContinuousMicrogridEnv.from_scenario(0, slack_module=('grid', 0))
    microgrid.set_module_attrs({'forecast_horizon': 0})
    reward_shaping_func = BaselineShaper()
    relative_reward_shaping_func = LearnedScaleRescaleShaper()

    import yaml
    print(yaml.safe_dump(reward_shaping_func))
    print(yaml.safe_dump(relative_reward_shaping_func))

    n_steps = 1000

    policy = ActionGenerator(microgrid, n_actions=n_steps)

    out = analyze_reward_shaping_func(microgrid, reward_shaping_func, policy=policy.reset(), n_steps=n_steps, plot=False)
    out_relative = analyze_reward_shaping_func(microgrid, relative_reward_shaping_func,
                                               policy=policy.reset(), n_steps=n_steps, plot=False)

    both = pd.concat([out[('balance', 0, 'shaped_reward')], out_relative[('balance', 0, 'shaped_reward')]],
                     keys=['absolute', 'relative'])

    sns.scatterplot(data=both, x='absolute', y='relative', marker='.')
    plt.show()

    excess_pv = out.loc[out['net_load'].squeeze() < 0]
