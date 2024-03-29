from copy import deepcopy
from math import factorial

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import DiscreteMicrogridEnv


class TestDiscreteEnv(TestCase):

    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv.from_microgrid(microgrid)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        # Add one for net load
        n_obs = 1 + sum([x.observation_space['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_init_from_modules(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv(microgrid.modules.to_tuples(), add_unbalanced_module=False)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        # Add one for net load
        n_obs = 1 + sum([x.observation_space['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))


class TestDiscreteEnvScenario(TestCase):
    microgrid_number = 0

    def setUp(self) -> None:
        self.env = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number)

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

        n_actions = factorial(n_action_modules) * (2 ** genset_modules)
        self.assertEqual(env.action_space.n, n_actions)

    def test_simple_observation_keys(self):
        keys_in_all_scenarios = ['load_current', 'renewable_current']

        env = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number,
                                                 observation_keys=keys_in_all_scenarios)

        obs, _, _, _ = env.step(env.action_space.sample())

        expected_obs = [
            env.modules['load'].item().state_dict(normalized=True)['load_current'],
            env.modules['pv'].item().state_dict(normalized=True)['renewable_current']
        ]

        self.assertEqual(obs.tolist(), expected_obs)

    def test_observation_keys_net_load_unnormalized(self):
        keys_in_all_scenarios = ['net_load']

        env = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number,
                                                 observation_keys=keys_in_all_scenarios)

        for j in range(3):
            with self.subTest(step=j):
                obs, _, _, _ = env.step(env.action_space.sample())

                load = env.modules['load'].item().state_dict(normalized=True)['load_current']
                renewable = env.modules['pv'].item().state_dict(normalized=True)['renewable_current']

                expected_obs = [(load-renewable) / load]

                self.assertEqual(obs.tolist(), expected_obs)

    def test_observation_keys_net_load_and_load_pv_unnormalized(self):
        keys_in_all_scenarios = ['renewable_current', 'net_load', 'load_current']

        env = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number,
                                                 observation_keys=keys_in_all_scenarios)

        for j in range(3):
            with self.subTest(step=j):
                obs, _, _, _ = env.step(env.action_space.sample())

                load = env.modules['load'].item().state_dict(normalized=True)['load_current']
                renewable = env.modules['pv'].item().state_dict(normalized=True)['renewable_current']

                expected_obs = [(load - renewable) / load, renewable, load]

                self.assertEqual(obs.tolist(), expected_obs)

    def test_set_initial_step(self):
        env = DiscreteMicrogridEnv.from_scenario(self.microgrid_number)
        env = deepcopy(env)

        self.assertEqual(env.initial_step, 0)

        self.assertEqual(env.initial_step, 0)
        self.assertEqual(env.modules.get_attrs('initial_step', unique=True, as_pandas=False), 0)

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
        self.assertEqual(env.modules.get_attrs('initial_step', unique=True, as_pandas=False), 1)

        for module_name, module_list in env.modules.iterdict():
            for n, module in enumerate(module_list):
                with self.subTest(module_name=module_name, module_num=n):
                    try:
                        initial_step = module.initial_step
                    except AttributeError:
                        continue

                    self.assertEqual(initial_step, 1)



class TestDiscreteEnvScenario1(TestDiscreteEnvScenario):
    microgrid_number = 1


class TestDiscreteEnvScenario2(TestDiscreteEnvScenario):
    microgrid_number = 2


class TestDiscreteEnvScenario3(TestDiscreteEnvScenario):
    microgrid_number = 3


class TestDiscreteEnvScenario4(TestDiscreteEnvScenario):
    microgrid_number = 4


class TestDiscreteEnvScenario5(TestDiscreteEnvScenario):
    microgrid_number = 5


class TestDiscreteEnvScenario6(TestDiscreteEnvScenario):
    microgrid_number = 6


class TestDiscreteEnvScenario47(TestDiscreteEnvScenario):
    microgrid_number = 7


class TestDiscreteEnvScenario8(TestDiscreteEnvScenario):
    microgrid_number = 8


class TestDiscreteEnvScenario9(TestDiscreteEnvScenario):
    microgrid_number = 9


class TestDiscreteEnvScenario10(TestDiscreteEnvScenario):
    microgrid_number = 10


class TestDiscreteEnvScenario11(TestDiscreteEnvScenario):
    microgrid_number = 11


class TestDiscreteEnvScenario12(TestDiscreteEnvScenario):
    microgrid_number = 12


class TestDiscreteEnvScenario13(TestDiscreteEnvScenario):
    microgrid_number = 13


class TestDiscreteEnvScenario14(TestDiscreteEnvScenario):
    microgrid_number = 14
    
    
class TestDiscreteEnvScenario15(TestDiscreteEnvScenario):
    microgrid_number = 15
    

class TestDiscreteEnvScenario16(TestDiscreteEnvScenario):
    microgrid_number = 16
    

class TestDiscreteEnvScenario17(TestDiscreteEnvScenario):
    microgrid_number = 17
    

class TestDiscreteEnvScenario18(TestDiscreteEnvScenario):
    microgrid_number = 18
    

class TestDiscreteEnvScenario19(TestDiscreteEnvScenario):
    microgrid_number = 19
    

class TestDiscreteEnvScenario20(TestDiscreteEnvScenario):
    microgrid_number = 20


class TestDiscreteEnvScenario21(TestDiscreteEnvScenario):
    microgrid_number = 21


class TestDiscreteEnvScenario22(TestDiscreteEnvScenario):
    microgrid_number = 22


class TestDiscreteEnvScenario23(TestDiscreteEnvScenario):
    microgrid_number = 23


class TestDiscreteEnvScenario24(TestDiscreteEnvScenario):
    microgrid_number = 24
