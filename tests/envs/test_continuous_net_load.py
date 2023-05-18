from copy import deepcopy
from math import factorial

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import NetLoadContinuousMicrogridEnv


class TestNetLoadContinuousEnv(TestCase):
    microgrid = get_modular_microgrid()
    env = NetLoadContinuousMicrogridEnv(microgrid)

    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = NetLoadContinuousMicrogridEnv(microgrid)

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
        # TODO write; check absolute
        pass

    def test_convert_action_to_relative(self):
        # TODO write; check relative
        pass


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
