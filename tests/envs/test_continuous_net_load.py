import numpy as np

from copy import deepcopy

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import NetLoadContinuousMicrogridEnv
from pymgrid.modules import RenewableModule


class TestNetLoadContinuousEnv(TestCase):
    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        n_obs = sum([x.observation_space['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_net_load(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        net_load = 10

        self.assertEqual(
            microgrid.modules.load.item().current_load-microgrid.modules.renewable.item().current_renewable, net_load)

        self.assertEqual(env.compute_net_load(), net_load)

        pass

    def test_convert_action_to_absolute(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        expected_absolute_action = {
            'battery': [np.array([5])],
            'genset': [np.array([0, 2.5])],
            'grid': [np.array([2.5])]
        }

        relative_action = np.array([0.5, 0, 0.25, 0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        expected_relative_action = np.array([0.5, 0, 0.25, 0.25])

        absolute_action = {
            'battery': [np.array([5])],
            'genset': [np.array([0, 2.5])],
            'grid': [np.array([2.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)

    def test_convert_action_to_absolute_zero_net_load(self):
        new_renewable_module = RenewableModule(time_series=60*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.compute_net_load(), 0.0)

        expected_absolute_action = {
            'battery': [np.array([0])],
            'genset': [np.array([0, 0])],
            'grid': [np.array([0])]
        }

        relative_action = np.array([0.5, 0, 0.25, 0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative_zero_net_load(self):
        new_renewable_module = RenewableModule(time_series=60 * np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.compute_net_load(), 0.0)

        expected_relative_action = np.array([0.0, 0.0, 0.0, 0.0])

        absolute_action = {
            'battery': [np.array([5])],
            'genset': [np.array([0, 2.5])],
            'grid': [np.array([2.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)

    def test_convert_action_to_absolute_negative_net_load(self):
        new_renewable_module = RenewableModule(time_series=70*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.compute_net_load(), -10.0)

        expected_absolute_action = {
            'battery': [np.array([-5])],
            'genset': [np.array([0, -2.5])],
            'grid': [np.array([-2.5])]
        }

        relative_action = np.array([0.5, 0, 0.25, 0.25])
        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative_negative_net_load(self):
        new_renewable_module = RenewableModule(time_series=70*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.compute_net_load(), -10.0)

        expected_relative_action = np.array([0.5, 0, 0.25, 0.25])

        absolute_action = {
            'battery': [np.array([-5])],
            'genset': [np.array([0, -2.5])],
            'grid': [np.array([-2.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)


class TestNetLoadContinuousEnvSlackModule(TestCase):
    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        self.assertEqual(env.slack_module, ('grid', 0))

        self.assertIn('battery', list(env._nested_action_space.keys()))
        self.assertIn('genset', list(env._nested_action_space.keys()))
        self.assertNotIn('grid', list(env._nested_action_space.keys()))

        n_obs = sum([x.observation_space['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_net_load(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        net_load = 10

        self.assertEqual(
            microgrid.modules.load.item().current_load - microgrid.modules.renewable.item().current_renewable, net_load)

        self.assertEqual(env.compute_net_load(), net_load)

    def test_convert_action_to_absolute(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        expected_absolute_action = {
            'battery': [np.array([5])],
            'genset': [np.array([0, 2.5])],
            'grid': [np.array([2.5])]
        }

        relative_action = np.array([0.5, 0, 0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        expected_relative_action = np.array([0.5, 0, 0.25])

        absolute_action = {
            'battery': [np.array([5])],
            'genset': [np.array([0, 2.5])],
            'grid': [np.array([2.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)

    def test_convert_action_to_absolute_zero_net_load(self):
        # TODO fix
        new_renewable_module = RenewableModule(time_series=60 * np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), 0.0)

        expected_absolute_action = {
            'battery': [np.array([0])],
            'genset': [np.array([0, 0])],
            'grid': [np.array([0])]
        }

        relative_action = np.array([0.5, 0, 0.25, 0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative_zero_net_load(self):
        # TODO fix
        new_renewable_module = RenewableModule(time_series=60 * np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), 0.0)

        expected_relative_action = np.array([0.0, 0.0, 0.0, 0.0])

        absolute_action = {
            'battery': [np.array([5])],
            'genset': [np.array([0, 2.5])],
            'grid': [np.array([2.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)

        # TODO write convert action tests for negative net load


class TestNetLoadContinuousEnvScenario(TestCase):
    microgrid_number = 0

    def setUp(self) -> None:
        self.env = NetLoadContinuousMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number)

    def test_run_once(self):
        env = deepcopy(self.env)
        # sample environment then get log
        self.assertEqual(len(env.log), 0)
        for j in range(10):
            with self.subTest(step=j):
                action = env.sample_action(strict_bound=True)
                env.step(action)
                self.assertEqual(len(env.log), j+1)

    def test_reset_after_run(self):
        env = deepcopy(self.env)
        env.step(env.sample_action(strict_bound=True))
        env.reset()
        self.assertEqual(len(env.log), 0)

    def test_run_again_after_reset(self):
        env = deepcopy(self.env)
        env.step(env.sample_action(strict_bound=True))

        self.assertEqual(len(env.log), 1)

        env.reset()

        self.assertEqual(len(env.log), 0)

        for j in range(10):
            with self.subTest(step=j):
                action = env.sample_action(strict_bound=True)
                env.step(action)
                self.assertEqual(len(env.log), j+1)

    def test_action_space(self):
        env = deepcopy(self.env)

        n_action_modules = len(env.modules.controllable.sources) + len(env.modules.controllable.source_and_sinks)
        genset_modules = len(env.modules.genset) if hasattr(env.modules, 'genset') else 0

        # TODO write
        return

    def test_simple_observation_keys(self):
        keys_in_all_scenarios = ['load_current', 'renewable_current']

        env = NetLoadContinuousMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number,
                                                 observation_keys=keys_in_all_scenarios)

        obs, _, _, _ = env.step(env.action_space.sample())

        expected_obs = [
            env.modules['load'].item().state_dict(normalized=True)['load_current'],
            env.modules['pv'].item().state_dict(normalized=True)['renewable_current']
        ]

        self.assertEqual(obs.tolist(), expected_obs)

    def test_set_initial_step(self):
        env = NetLoadContinuousMicrogridEnv.from_scenario(self.microgrid_number)
        env = deepcopy(env)

        self.assertEqual(env.initial_step, 0)

        self.assertEqual(env.initial_step, 0)
        self.assertEqual(
            env.modules.get_attrs('initial_step', unique=True, as_pandas=False),
            {'initial_step': 0}
        )

        for module_name, module_list in env.modules.iterdict():
            for n, module in enumerate(module_list):
                with self.subTest(module_name=module_name, module_num=n):
                    try:
                        initial_step = module.initial_step
                    except AttributeError:
                        continue

                    self.assertEqual(initial_step, 0)

        env = deepcopy(env)

        env.initial_step = 1

        self.assertEqual(env.initial_step, 1)
        self.assertEqual(
            env.modules.get_attrs('initial_step', unique=True, as_pandas=False),
            {'initial_step': 1}
        )

        for module_name, module_list in env.modules.iterdict():
            for n, module in enumerate(module_list):
                with self.subTest(module_name=module_name, module_num=n):
                    try:
                        initial_step = module.initial_step
                    except AttributeError:
                        continue

                    self.assertEqual(initial_step, 1)
