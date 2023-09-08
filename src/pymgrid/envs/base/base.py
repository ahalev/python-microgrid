import numpy as np
import pandas as pd

from collections import OrderedDict
from gym import Env
from gym.spaces import Box, Dict, Tuple, flatten_space, flatten
from abc import abstractmethod

from pymgrid import NonModularMicrogrid, Microgrid
from pymgrid.envs.base.skip_init import skip_init


class BaseMicrogridEnv(Microgrid, Env):
    """
    Base class for all microgrid environments.

    Implements the `OpenAI Gym API <https://www.gymlibrary.dev//>`_ for a microgrid;
    inherits from both :class:`.Microgrid` and :class:`gym.Env`.

    Parameters
    ----------
    modules : list, Microgrid, NonModularMicrogrid, or int.
        The constructor can be called in three ways:

        1. Passing a list of microgrid modules. This is identical to the :class:`.Microgrid` constructor.

        2. Passing a :class:`.Microgrid` or :class:`.NonModularMicrogrid` instance.
           This will effectively wrap the microgrid instance with the Gym API.

        3. Passing an integer in [0, 25).
           This will be result in loading the corresponding `pymgrid25` benchmark microgrids.

    add_unbalanced_module : bool, default True.
        Whether to add an unbalanced energy module to your microgrid. Such a module computes and attributes
        costs to any excess supply or demand.
        Set to True unless ``modules`` contains an :class:`.UnbalancedEnergyModule`.

    loss_load_cost : float, default 10.0
        Cost per unit of unmet demand. Ignored if ``add_unbalanced_module=False``.

    overgeneration_cost : float, default 2.0
        Cost per unit of excess generation.  Ignored if ``add_unbalanced_module=False``.

    flat_spaces : bool, default True
        Whether the environment's spaces should be flat.

        If True, all continuous spaces are :class:`gym:gym.spaces.Box`.

        Otherwise, they are nested :class:`gym:gym.spaces.Dict` of :class:`gym:gym.spaces.Tuple`
        of :class:`gym:gym.spaces.Box`, corresponding to the structure of the ``control`` arg of :meth:`.Microgrid.run`.

    trajectory_func : callable or None, default None
        Callable that sets an initial and final step for an episode. ``trajectory_func`` must take two inputs:
        :attr:`.initial_step` and :attr:`.final_step`, and return two integers: the initial and final step for
        that particular episode, respectively. This function will be called every time :meth:`.reset` is called.

        If None, :attr:`.initial_step` and :attr:`.final_step` are used to define every episode.

    step_callback: callable or None, default None
        Function to call on every ``step``.

    reset_callback: callable or None, default None
        Function to call on every ``reset``.

    """

    action_space = None
    'Space object corresponding to valid actions.'

    observation_space = None
    'Space object corresponding to valid observations.'

    def __new__(cls, modules, *args, **kwargs):
        if isinstance(modules, (NonModularMicrogrid, Microgrid)):
            import warnings
            warnings.warn('Initializing an environment with a microgrid will be deprecated in a future version.'
                          'Use from_microgrid() instead.', category=FutureWarning)

            instance = cls.from_microgrid(modules, **kwargs)

        elif isinstance(modules, int):
            import warnings
            warnings.warn('Initializing an environment with a scenario integer will be deprecated in a future version.'
                          'Use from_scenario() instead.', category=FutureWarning)
            instance = cls.from_scenario(modules, **kwargs)

        else:
            return super().__new__(cls)

        cls.__init__ = skip_init(cls, cls.__init__)
        return instance

    def __init__(self,
                 modules,
                 add_unbalanced_module=True,
                 loss_load_cost=10,
                 overgeneration_cost=2,
                 reward_shaping_func=None,
                 trajectory_func=None,
                 flat_spaces=True,
                 observation_keys=(),
                 step_callback=None,
                 reset_callback=None
                 ):

        super().__init__(modules,
                         add_unbalanced_module=add_unbalanced_module,
                         loss_load_cost=loss_load_cost,
                         overgeneration_cost=overgeneration_cost,
                         reward_shaping_func=reward_shaping_func,
                         trajectory_func=trajectory_func)

        self._flat_spaces = flat_spaces
        self.observation_keys = self._validate_observation_keys(observation_keys)
        self.step_callback = step_callback if step_callback is not None else lambda *a, **k: None
        self.reset_callback = reset_callback if reset_callback is not None else lambda *a, **k: None

        self.action_space = self._get_action_space()
        self.observation_space, self._nested_observation_space = self._get_observation_space()

    def _validate_observation_keys(self, keys):
        if not keys:
            return keys

        if isinstance(keys, str):
            keys = [keys]

        keys = np.array(keys)

        possible_keys = self.potential_observation_keys()
        bad_keys = [key for key in keys if key not in possible_keys]

        if bad_keys:
            raise NameError(f'Keys {bad_keys} not found in state.')

        # Put net load at the beginning, to match where it will be in the action space
        net_load_pos = np.where(np.array(keys) == 'net_load')[0]

        if net_load_pos.size:
            keys[[0, net_load_pos.item()]] = keys[[net_load_pos.item(), 0]]

        return keys.tolist()

    @abstractmethod
    def _get_action_space(self, remove_redundant_actions=False):
        pass

    def _get_observation_space(self):
        obs_space = {}

        if self.observation_keys and 'net_load' in self.observation_keys:
            obs_space['net_load'] = Tuple([Box(low=-np.inf, high=1, shape=(1, ), dtype=np.float64)])

        state_series = self.state_series()

        for name, module_list in self.modules.iterdict():
            tup = []
            for module_num, module in enumerate(module_list):
                normalized_space = module.observation_space['normalized']

                if not self.observation_keys:
                    tup.append(normalized_space)
                else:
                    try:
                        relevant_state_idx = state_series.loc[pd.IndexSlice[name, module_num]].index
                    except KeyError:
                        continue

                    locs = [
                        relevant_state_idx.get_loc(key) for key in self.observation_keys if key in relevant_state_idx
                    ]
                    if locs:
                        box_slice = Box(
                            normalized_space.low[locs],
                            normalized_space.high[locs],
                            shape=(len(locs), ),
                            dtype=normalized_space.dtype
                        )

                        tup.append(box_slice)
            if tup:
                obs_space[name] = Tuple(tup)

        # Prevent sorting of keys; first cast to OrderedDict
        obs_space = Dict(OrderedDict(obs_space))

        return (flatten_space(obs_space) if self._flat_spaces else obs_space), obs_space

    def potential_observation_keys(self):
        return self.state_series().index.get_level_values(-1).unique()

    def reset(self):
        obs = super().reset()
        obs.pop('balance')
        obs.pop('other')
        self.reset_callback()
        return self._get_obs(obs)

    def step(self, action, normalized=True):
        """
        Run one timestep of the environment's dynamics.

        When the end of the episode is reached, you are responsible for calling `reset()`
        to reset the environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action : int or np.ndarray
            An action provided by the agent.

        normalized : bool, default True
            Whether the passed action is normalized or not.

        Returns
        -------
        observation : dict[str, list[float]] or np.ndarray, shape self.observation_space.shape
            Observations of each module after using the passed ``action``.
            ``observation`` is a nested dict if :attr:`~.flat_spaces` is True and a one-dimensional numpy array
            otherwise.

        reward : float
            Reward/cost of running the microgrid. A positive value implies revenue while a negative
            value is a cost.

        done : bool
            Whether the microgrid terminates.

        info : dict
            Additional information from this step.

        """
        self._microgrid_logger.log(net_load=self.compute_net_load())

        action = self.convert_action(action)
        self._log_action(action, normalized)

        obs, reward, done, info = self.run(action, normalized=normalized)
        obs = self._get_obs(obs)
        self.step_callback(**self._get_step_callback_info(action, obs, reward, done, info))

        return obs, reward, done, info

    @abstractmethod
    def convert_action(self, action):
        """
        Convert a reinforcement learning action to a microgrid control.

        In a discrete environment, for example, converts an integer to a microgrid control.

        Parameters
        ----------
        action : int, np.ndarray or dict
            Action to convert. Integer if discrete, np.ndarray if continuous,
            dict if converting from a microgrid action.

        Returns
        -------
        converted_action : dict[str, list[float]]
            Resultant microgrid control.
        """
        pass

    def _log_action(self, action, normalized, log_column='converted_action'):
        d = {}

        log_items = [(log_column, action)]

        if normalized:
            log_items.append((f'denormalized_{log_column}', self.microgrid_action_space.denormalize(action)))

        for key, action in log_items:
            for module, action_list in action.items():
                for j, act in enumerate(action_list):
                    if not pd.api.types.is_list_like(act):
                        act = [act]
                    d.update({(key, j, f'{module}_{el_num}'): act_n for el_num, act_n in enumerate(act)})

        self._microgrid_logger.log(d)

    def _get_obs(self, obs):
        if self.observation_keys and self._flat_spaces:
            _obs = np.array([module_state[k] for k in self.observation_keys
                               for module_name, module_states in self.state_dict(normalized=True).items()
                               for module_state in module_states if k in module_state])

            if np.random.rand() < 0.01:
                # This is an assertion to check that this is the same as the old computation
                # Can be removed at a later date
                assert np.isclose(_obs, self._get_obs_old(obs)).all()

            return _obs

        return self._get_obs_old(obs)

    def _get_obs_old(self, obs):
        if self.observation_keys:
            obs = self.state_series(normalized=True).loc[pd.IndexSlice[:, :, self.observation_keys]]

            if self._flat_spaces:
                obs = obs.values

            else:
                obs = obs.to_frame().unstack(level=1).T.droplevel(level=1, axis=1).to_dict(orient='list')

        elif self._flat_spaces:
            obs = self.flatten_obs(self._nested_observation_space, obs)

        return obs

    def _get_step_callback_info(self, action, obs, reward, done, info):
        return {
            'action': action,
            'obs': obs,
            'reward': reward,
            'done': done,
            'info': info
        }

    def render(self, mode="human"):
        """:meta private:"""
        raise RuntimeError('rendering is not possible in Microgrid environments.')

    def sample_action(self, strict_bound=False, sample_flex_modules=False):
        return self.action_space.sample()

    def compute_net_load(self, normalized=False):
        """
        Compute the net load at the current step.

        Net load is load minus renewables.
        -------

        Returns
        -------
        net_load : float
            Net load.

        """

        try:
            fixed_consumption = self.modules.fixed.get_attrs('max_consumption', as_pandas=False, drop_attr_names=True)
            fixed_consumption = np.sum(list(fixed_consumption.values()))
        except AttributeError:
            fixed_consumption = 0.0
        except IndexError:
            # Exhausted available data. Episode should be over
            assert self.current_step == self.final_step
            return 0.0

        try:
            flex_max_prod = [m.max_production for m in self.modules.flex.iterlist() if m.marginal_cost == 0]
        except IndexError:
            # Exhausted available data. Episode should be over
            assert self.current_step == self.final_step
            return 0.0

        flex_production = sum(flex_max_prod)

        net_load = fixed_consumption - flex_production

        if normalized:
            if fixed_consumption:
                return net_load / fixed_consumption
            return -1.0

        return net_load

    def state_dict(self, normalized=False, as_run_output=False):
        sd = super().state_dict(normalized=normalized, as_run_output=as_run_output)

        net_load = self.compute_net_load(normalized=normalized)

        if as_run_output:
            net_load_entry = np.array([net_load])
        else:
            net_load_entry = {'net_load': net_load}

        sd['general'] = [net_load_entry]

        return sd

    @staticmethod
    def flatten_obs(observation_space, obs):
        return np.concatenate([flatten(observation_space[k], v) for k, v in obs.items()])

    @property
    def unwrapped(self):
        """:meta private:"""
        return super().unwrapped

    @property
    def flat_spaces(self):
        """
        Whether the environment's spaces are flat.

        If True, all continuous spaces are :class:`gym:gym.spaces.Box`.

        Otherwise, they are nested :class:`gym:gym.spaces.Dict` of :class:`gym:gym.spaces.Tuple`
        of :class:`gym:gym.spaces.Box`, corresponding to the structure of the ``control`` arg of :meth:`Microgrid.run`.

        Returns
        -------
        flat_spaces : bool
            Whether the environment's spaces are flat.

        """
        return self._flat_spaces

    @classmethod
    def from_microgrid(cls, microgrid, **kwargs):
        """
        Construct an RL environment from a microgrid.

        Effectively wraps the microgrid with the environment API.

        .. warning::
            Any logs contained in the microgrid will not be ported over to the environment.

        Parameters
        ----------
        microgrid : :class:`pymgrid.Microgrid`
            Microgrid to wrap.

        Returns
        -------
        env
            The environment, suitable for reinforcement learning.

        """
        try:
            modules = microgrid.modules
        except AttributeError:
            assert isinstance(microgrid, NonModularMicrogrid)
            return cls.from_nonmodular(microgrid, **kwargs)

        kwargs = kwargs.copy()

        kwargs['add_unbalanced_module'] = kwargs.pop('add_unbalanced_module', False)
        kwargs['reward_shaping_func'] = kwargs.pop('reward_shaping_func', microgrid.reward_shaping_func)
        kwargs['trajectory_func'] = kwargs.pop('trajectory_func', microgrid.trajectory_func)

        return cls(modules.to_tuples(), **kwargs)

    @classmethod
    def from_nonmodular(cls, nonmodular, **kwargs):
        microgrid = super().from_nonmodular(nonmodular)
        return cls.from_microgrid(microgrid, **kwargs)

    @classmethod
    def from_scenario(cls, microgrid_number=0, **kwargs):
        env = super().from_scenario(microgrid_number=microgrid_number)

        if kwargs:
            return cls.from_microgrid(env, **kwargs)

        return env

    @classmethod
    def load(cls, stream):
        return cls(super().load(stream))
