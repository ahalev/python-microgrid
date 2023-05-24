import numpy as np

from copy import deepcopy
from gym.spaces import Box

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from tests.envs.test_discrete import TestDiscreteEnvScenario

from pymgrid.envs import NetLoadContinuousMicrogridEnv
from pymgrid.modules import RenewableModule
from pymgrid import Microgrid


class TestNetLoadContinuousEnv(TestCase):
    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        n_obs = sum([x.observation_space['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_action_space(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        n_actions = len(env.modules.controllable)

        if 'genset' in env.modules:
            n_actions += len(env.modules.genset)

        self.assertEqual(env.action_space, Box(low=0, high=1, shape=(n_actions, )))

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

    def test_convert_action_to_absolute_negative_net_load_clip_actions(self):
        new_renewable_module = RenewableModule(time_series=70*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.compute_net_load(), -10.0)

        expected_absolute_action = {
            'battery': [np.array([-5])],
            'genset': [np.array([0, 0.0])],
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

    def test_convert_action_to_absolute_negative_net_load_no_clip_actions(self):
        new_renewable_module = RenewableModule(time_series=70*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, clip_actions=False)

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


class TestNetLoadContinuousEnvSlackModule(TestCase):
    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        self.assertEqual(env.slack_module, ('grid', 0))

        try:
            action_space_keys = list(env._nested_action_space.keys())
        except AttributeError: # Dict object does not subclass mapping in old version of gym
            action_space_keys = list(env._nested_action_space.spaces.keys())

        self.assertIn('battery', action_space_keys)
        self.assertIn('genset', action_space_keys)
        self.assertNotIn('grid', action_space_keys)

        n_obs = sum([x.observation_space['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_action_space(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        n_actions = len(env.modules.controllable) - 1 # subtract grid, not in action space

        if 'genset' in env.modules:
            n_actions += len(env.modules.genset)

        self.assertEqual(env.action_space, Box(low=0, high=1, shape=(n_actions, )))

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
        new_renewable_module = RenewableModule(time_series=60 * np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), 0.0)

        expected_absolute_action = {
            'battery': [np.array([0])],
            'genset': [np.array([0, 0])],
            'grid': [np.array([0])]
        }

        relative_action = np.array([0.5, 0, 0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative_zero_net_load(self):
        new_renewable_module = RenewableModule(time_series=60 * np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), 0.0)

        expected_relative_action = np.array([0.0, 0.0, 0.0])

        absolute_action = {
            'battery': [np.array([5])],
            'genset': [np.array([0, 2.5])],
            'grid': [np.array([2.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)

    def test_convert_action_to_absolute_negative_net_load_with_clip(self):
        new_renewable_module = RenewableModule(time_series=70*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), -10.0)

        expected_absolute_action = {
            'battery': [np.array([-5.])],
            'genset': [np.array([0, 0])],
            'grid': [np.array([-5.])]
        }

        relative_action = np.array([0.5, 0, 0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_absolute_negative_net_load_no_clip(self):
        new_renewable_module = RenewableModule(time_series=70*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0), clip_actions=False)

        self.assertEqual(env.compute_net_load(), -10.0)

        expected_absolute_action = {
            'battery': [np.array([-5.])],
            'genset': [np.array([0, -2.5])],
            'grid': [np.array([-2.5])]
        }

        relative_action = np.array([0.5, 0, 0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative_negative_net_load(self):
        new_renewable_module = RenewableModule(time_series=70*np.ones(100))
        microgrid = get_modular_microgrid(remove_modules=['renewable'], additional_modules=[new_renewable_module])

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), -10.0)

        expected_relative_action = np.array([0.5, 0, 0.25])

        absolute_action = {
            'battery': [np.array([-5])],
            'genset': [np.array([0, -2.5])],
            'grid': [np.array([-2.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)

    def test_convert_action_to_absolute_different_signs_with_clip(self):
        microgrid = get_modular_microgrid()

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), 10.0)

        expected_absolute_action = {
            'battery': [np.array([5.0])],
            'genset': [np.array([1, 0.0])],
            'grid': [np.array([5.0])]
        }

        relative_action = np.array([0.5, 1, -0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_absolute_different_signs_no_clip(self):
        microgrid = get_modular_microgrid()

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0), clip_actions=False)

        self.assertEqual(env.compute_net_load(), 10.0)

        expected_absolute_action = {
            'battery': [np.array([5.0])],
            'genset': [np.array([1, -2.5])],
            'grid': [np.array([7.5])]
        }

        relative_action = np.array([0.5, 1, -0.25])

        absolute_action = env.convert_action(relative_action)

        for module_name, action_list in expected_absolute_action.items():
            for module_num, act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    self.assertEqual(act, absolute_action[module_name][module_num])

    def test_convert_action_to_relative_different_signs(self):
        microgrid = get_modular_microgrid()

        env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=('grid', 0))

        self.assertEqual(env.compute_net_load(), 10.0)

        expected_relative_action = np.array([0.5, 1, -0.25])


        absolute_action = {
            'battery': [np.array([5.0])],
            'genset': [np.array([1, -2.5])],
            'grid': [np.array([7.5])]
        }

        relative_action = env.convert_action(absolute_action, to_microgrid=False)
        self.assertEqual(relative_action, expected_relative_action)


class TestNetLoadContinuousEnvScenario(TestDiscreteEnvScenario):
    microgrid_number = 0

    def setUp(self) -> None:
        self.env = NetLoadContinuousMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number)

    def test_action_space(self):
        from gym.spaces import Box

        env = deepcopy(self.env)

        controllable = len(env.modules.controllable)
        genset_modules = len(env.modules.genset) if hasattr(env.modules, 'genset') else 0

        action_dim = controllable + genset_modules

        self.assertEqual(env.action_space, Box(low=0, high=1, shape=(action_dim, )))


class TestNetLoadContinuousEnvScenario1(TestNetLoadContinuousEnvScenario):
    microgrid_number = 1


class TestNetLoadContinuousEnvScenario2(TestNetLoadContinuousEnvScenario):
    microgrid_number = 2


class TestNetLoadContinuousEnvScenario3(TestNetLoadContinuousEnvScenario):
    microgrid_number = 3


class TestNetLoadContinuousEnvScenario4(TestNetLoadContinuousEnvScenario):
    microgrid_number = 4


class TestNetLoadContinuousEnvScenario5(TestNetLoadContinuousEnvScenario):
    microgrid_number = 5


class TestNetLoadContinuousEnvScenario6(TestNetLoadContinuousEnvScenario):
    microgrid_number = 6


class TestNetLoadContinuousEnvScenario7(TestNetLoadContinuousEnvScenario):
    microgrid_number = 7


class TestNetLoadContinuousEnvScenario8(TestNetLoadContinuousEnvScenario):
    microgrid_number = 8


class TestNetLoadContinuousEnvScenario9(TestNetLoadContinuousEnvScenario):
    microgrid_number = 9


class TestNetLoadContinuousEnvScenario10(TestNetLoadContinuousEnvScenario):
    microgrid_number = 10


class TestNetLoadContinuousEnvScenario11(TestNetLoadContinuousEnvScenario):
    microgrid_number = 11


class TestNetLoadContinuousEnvScenario12(TestNetLoadContinuousEnvScenario):
    microgrid_number = 12


class TestNetLoadContinuousEnvScenario13(TestNetLoadContinuousEnvScenario):
    microgrid_number = 13


class TestNetLoadContinuousEnvScenario14(TestNetLoadContinuousEnvScenario):
    microgrid_number = 14


class TestNetLoadContinuousEnvScenario15(TestNetLoadContinuousEnvScenario):
    microgrid_number = 15


class TestNetLoadContinuousEnvScenario16(TestNetLoadContinuousEnvScenario):
    microgrid_number = 16


class TestNetLoadContinuousEnvScenario17(TestNetLoadContinuousEnvScenario):
    microgrid_number = 17


class TestNetLoadContinuousEnvScenario18(TestNetLoadContinuousEnvScenario):
    microgrid_number = 18


class TestNetLoadContinuousEnvScenario19(TestNetLoadContinuousEnvScenario):
    microgrid_number = 19


class TestNetLoadContinuousEnvScenario20(TestNetLoadContinuousEnvScenario):
    microgrid_number = 20


class TestNetLoadContinuousEnvScenario21(TestNetLoadContinuousEnvScenario):
    microgrid_number = 21


class TestNetLoadContinuousEnvScenario22(TestNetLoadContinuousEnvScenario):
    microgrid_number = 22


class TestNetLoadContinuousEnvScenario23(TestNetLoadContinuousEnvScenario):
    microgrid_number = 23


class TestNetLoadContinuousEnvScenario24(TestNetLoadContinuousEnvScenario):
    microgrid_number = 24


class TestNetLoadContinuousEnvSlackScenario(TestDiscreteEnvScenario):
    microgrid_number = 0

    def setUp(self) -> None:
        microgrid = Microgrid.from_scenario(self.microgrid_number)
        self.slack_module = ('grid', 0) if hasattr(microgrid.modules, 'grid') else ('genset', 0)

        self.env = NetLoadContinuousMicrogridEnv.from_microgrid(microgrid, slack_module=self.slack_module)

    def test_module_existence(self):
        try:
            grid = self.env.modules.grid.item()
        except AttributeError:
            pass
        else:
            if grid.weak_grid:
                self.assertTrue(hasattr(self.env.modules, 'genset'))

            if hasattr(self.env.modules, 'genset'):
                self.assertLessEqual(grid.marginal_cost, self.env.modules.genset.item().marginal_cost)


    def test_action_space(self):
        from gym.spaces import Box

        env = deepcopy(self.env)

        controllable = len(env.modules.controllable)
        genset_modules = len(env.modules.genset) if hasattr(env.modules, 'genset') else 0

        action_dim = controllable + genset_modules

        if 'genset' in self.slack_module:
            action_dim -= 2
        else:
            action_dim -= 1

        self.assertEqual(env.action_space, Box(low=0, high=1, shape=(action_dim, )))


class TestNetLoadContinuousEnvSlackScenario2(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 2


class TestNetLoadContinuousEnvSlackScenario3(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 3


class TestNetLoadContinuousEnvSlackScenario4(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 4


class TestNetLoadContinuousEnvSlackScenario5(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 5


class TestNetLoadContinuousEnvSlackScenario6(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 6


class TestNetLoadContinuousEnvSlackScenario7(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 7


class TestNetLoadContinuousEnvSlackScenario8(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 8


class TestNetLoadContinuousEnvSlackScenario9(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 9


class TestNetLoadContinuousEnvSlackScenario10(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 10


class TestNetLoadContinuousEnvSlackScenario11(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 11


class TestNetLoadContinuousEnvSlackScenario12(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 12


class TestNetLoadContinuousEnvSlackScenario13(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 13


class TestNetLoadContinuousEnvSlackScenario14(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 14


class TestNetLoadContinuousEnvSlackScenario15(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 15


class TestNetLoadContinuousEnvSlackScenario16(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 16


class TestNetLoadContinuousEnvSlackScenario17(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 17


class TestNetLoadContinuousEnvSlackScenario18(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 18


class TestNetLoadContinuousEnvSlackScenario19(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 19


class TestNetLoadContinuousEnvSlackScenario20(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 20


class TestNetLoadContinuousEnvSlackScenario21(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 21


class TestNetLoadContinuousEnvSlackScenario22(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 22


class TestNetLoadContinuousEnvSlackScenario23(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 23


class TestNetLoadContinuousEnvSlackScenario24(TestNetLoadContinuousEnvSlackScenario):
    microgrid_number = 24
