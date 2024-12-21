import functools
import numpy as np
import pandas as pd

from copy import deepcopy

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import DiscreteMicrogridEnv, ContinuousMicrogridEnv, NetLoadContinuousMicrogridEnv
from pymgrid.envs.base import BaseMicrogridEnv


def pass_if_parent(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.env_class is not None:
            return func(self, *args, **kwargs)

    return wrapper


class Parent(TestCase):
    env_class: BaseMicrogridEnv = None
    observation_keys = ()

    @pass_if_parent
    def setUp(self) -> None:
        self.env = self.env_class.from_microgrid(get_modular_microgrid(), observation_keys=self.observation_keys)

    @pass_if_parent
    def test_reset_obs_in_obs_space(self):
        env = deepcopy(self.env)
        obs = env.reset()

        self.assertIn(obs, env.observation_space)

    @pass_if_parent
    def test_pre_reset_state_series_invariant_to_observation_keys(self):
        env = deepcopy(self.env)

        self.assertEqual(env.state_series().shape, (13, ))

    @pass_if_parent
    def test_flattened_state_dict_is_state_series(self):
        env = deepcopy(self.env)

        state_dict = env.state_dict(normalized=True, as_run_output=True)
        flattened_state_dict = flatten_nested_dict(state_dict)

        self.assertEqual(flattened_state_dict, env.state_series(normalized=True).values)

    @pass_if_parent
    def test_pre_reset_state_dict_invariant_to_observation_keys(self):
        env = deepcopy(self.env)
        state_dict = env.state_dict()

        n_state_dict_values = functools.reduce(lambda x, y: x+len(y[0]), state_dict.values(), 0)

        self.assertEqual(n_state_dict_values, 13)

    @pass_if_parent
    def test_obs_values_after_reset(self):
        env = deepcopy(self.env)
        obs = env.reset()

        if self.observation_keys:
            expected_obs = env.state_series(normalized=True).loc[pd.IndexSlice[:, :, self.observation_keys]].values
        else:
            expected_obs = env.state_series(normalized=True).values

        self.assertEqual(obs, expected_obs)

    @pass_if_parent
    def test_state_series_values(self):
        env = deepcopy(self.env)

        expected_state_series = np.array([10., -60., 50., 1., 1., 0., 0., 0.5, 50., 1., 1., 1., 1.])
        self.assertEqual(env.state_series(normalized=False).values, expected_state_series)

    @pass_if_parent
    def test_state_series_values_normalized(self):
        env = deepcopy(self.env)

        expected_state_series = np.array([1/6., 0., 1., 1., 1., 0., 0., 0.5, 0.5, 0., 0., 0., 0.])
        self.assertEqual(env.state_series(normalized=True).values, expected_state_series)

    @pass_if_parent
    def test_get_obs(self):
        env = deepcopy(self.env)
        obs = env._get_obs()

        if self.observation_keys:
            expected_obs = env.state_series(normalized=True).loc[pd.IndexSlice[:, :, self.observation_keys]].values
        else:
            expected_obs = env.state_series(normalized=True).values

        self.assertEqual(obs, expected_obs)

    @pass_if_parent
    def test_steps(self):
        env = deepcopy(self.env)

        env.reset()

        for j in range(10):
            with self.subTest(step=j):
                obs, _, _, _ = env.step(env.action_space.sample())
                self.assertTrue((obs >= 0).all())
                self.assertTrue((obs <= 1).all())


class ObsKeysNoNetLoadParent(Parent):
    observation_keys = ['soc', 'import_price_current', 'goal_status', 'load_current', 'renewable_current']

    @pass_if_parent
    def test_get_obs_correct_keys_in_modules(self):
        env = deepcopy(self.env)
        obs = env._get_obs()

        for module in env.modules.iterlist():
            module_state_dict = module.state_dict(normalized=True)
            matching_keys = [obs_key for obs_key in self.observation_keys if obs_key in module.state_dict().keys()]
            matching_values = [module_state_dict[k] for k in matching_keys]

            with self.subTest(module=module.name, keys=matching_keys):
                self.assertEqual(obs[np.isin(self.observation_keys, matching_keys)], matching_values)


class ObsKeysWithNetLoadParent(ObsKeysNoNetLoadParent):
    observation_keys = ['net_load', 'soc', 'load_current', 'export_price_current']


class ObsKeysDuplicateKeysParent(ObsKeysNoNetLoadParent):
    observation_keys = ['net_load', 'soc', 'load_current', 'load_current', 'export_price_current']

    def test_get_obs_correct_keys_in_modules(self):
        env = deepcopy(self.env)
        obs = env._get_obs()

        unique_obs_keys = pd.Index(self.observation_keys).drop_duplicates().tolist()

        for module in env.modules.iterlist():
            module_state_dict = module.state_dict(normalized=True)
            matching_keys = [obs_key for obs_key in unique_obs_keys if obs_key in module.state_dict().keys()]
            matching_values = [module_state_dict[k] for k in matching_keys]

            with self.subTest(module=module.name, keys=matching_keys):
                self.assertEqual(obs[np.isin(unique_obs_keys, matching_keys)], matching_values)


class TestDiscrete(Parent):
    env_class = DiscreteMicrogridEnv


class TestContinuous(Parent):
    env_class = ContinuousMicrogridEnv


class TestNetLoadContinuous(Parent):
    env_class = NetLoadContinuousMicrogridEnv


class TestDiscreteObsKeysNoNetLoad(ObsKeysNoNetLoadParent):
    env_class = DiscreteMicrogridEnv


class TestContinuousObsKeysNoNetLoad(ObsKeysNoNetLoadParent):
    env_class = ContinuousMicrogridEnv


class TestNetLoadContinuousObsKeysNoNetLoad(ObsKeysNoNetLoadParent):
    env_class = NetLoadContinuousMicrogridEnv


class TestDiscreteObsDuplicateKeys(ObsKeysDuplicateKeysParent):
    env_class = DiscreteMicrogridEnv


class TestContinuousObsDuplicateKeys(ObsKeysDuplicateKeysParent):
    env_class = ContinuousMicrogridEnv


class TestNetLoadContinuousObsDuplicateKeys(ObsKeysDuplicateKeysParent):
    env_class = NetLoadContinuousMicrogridEnv


def flatten_nested_dict(nested_dict):
    def extract_list(l):
        assert len(l) == 1, 'reduction only works with length 1 lists'
        return l[0].tolist()

    return functools.reduce(lambda x, y: x + extract_list(y), nested_dict.values(), [])
