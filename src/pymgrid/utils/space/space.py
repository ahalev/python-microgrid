import operator
import numpy as np
import warnings

from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from gym.spaces import Box, Dict, Space, Tuple
from typing import Union


def _extract_action_spaces(d):
    controllable = {}
    for module_name, module_list in d.items():
        try:
            module_list = module_list.to_dict()  # module_list is pd.Series
            controllable_spaces = [act_space for j, act_space in enumerate(module_list['action_space'])
                                   if 'controllable' in module_list['module_type'][j]]
        except AttributeError:
            controllable_spaces = [v['action_space'] for v in module_list if 'controllable' in v['module_type']]

        if controllable_spaces:
            controllable[module_name] = controllable_spaces

    return controllable


def _extract_observation_spaces(d):
    obs_spaces = {}
    for module_name, module_list in d.items():
        try:
            module_list = module_list.to_dict()  # module_list is pd.Series
            spaces = module_list['observation_space']
        except AttributeError:
            spaces = [v['observation_space'] for v in module_list]

        if spaces:
            obs_spaces[module_name] = spaces

    return obs_spaces


def _transform_builtins(d, normalized=False):
    space_key = 'normalized' if normalized else 'unnormalized'

    transformed = {}

    if isinstance(d, dict):
        transformed = {}
        for k, v in d.items():
            if isinstance(v, _PymgridSpace):
                transformed[k] = v[space_key]
            else:
                transformed[k] = _transform_builtins(v, normalized=normalized)

        transformed = Dict(transformed)

    elif isinstance(d, list):
        transformed = []
        for v in d:
            if isinstance(v, _PymgridSpace):
                transformed.append(v[space_key])
            elif isinstance(v, Space):
                transformed.append(v)
            else:
                transformed.append(_transform_builtins(v, normalized=normalized))

        transformed = Tuple(transformed)

    return transformed


def extract_builtins(d, act_or_obs='act', normalized=False):
    try:
        d = d.groupby(level=0, axis=0).agg(list).T
    except AttributeError:
        pass

    if act_or_obs == 'act':
        spaces = _extract_action_spaces(d)
    elif act_or_obs == 'obs':
        spaces = _extract_observation_spaces(d)
    else:
        raise NameError(act_or_obs)

    return _transform_builtins(spaces, normalized=normalized)


class _PymgridDict(Dict):
    def __init__(self, d, seed=None):
        try:
            super().__init__(d, seed=seed)
        except AssertionError:
            import gym
            warnings.warn(f"gym.Space does not accept argument 'seed' in version {gym.__version__}; this argument will "
                          f"be ignored. Upgrade your gym version with 'pip install -U gym' to use this functionality.")

            super().__init__(d.spaces)

    def get_attr(self, attr):
        return {k: [getattr(v, attr) for v in tup] for k, tup in self.items()}

    @property
    def low(self):
        return self.get_attr('low')

    @property
    def high(self):
        return self.get_attr('high')

    @property
    def shape(self):
        return self.get_attr('shape')

    @shape.setter
    def shape(self, value):
        """
        Necessary for compatability with gym<0.21.
        """
        assert value is None

    def __getattr__(self, item):
        """
        Necessary for compatability with gym<0.21, where gym.spaces.Dict did not inherit from collections.Mapping.
        """
        if item == 'spaces':
            raise AttributeError('spaces')
        try:
            return getattr(self.spaces, item)
        except AttributeError:
            raise AttributeError(item)


class _PymgridSpace(Space):
    _unnormalized: Union[Box, _PymgridDict]
    _normalized: Union[Box, _PymgridDict]
    _spread: Union[np.ndarray, dict]

    def contains(self, x):
        """
        Check if `x` is a valid member of the space.

        .. note::
            This method checks if `x` is a valid member of the space. Use :meth:`.ModuleSpace.normalized.contains` to
            check if a value is a member of the normalized space.

        Parameters
        ----------
        x : scalar or array-like
            Value to check membership.

        Returns
        -------
        containment : bool
            Whether `x` is a valid member of the space.

        """
        return x in self._unnormalized

    def sample(self, normalized=False):
        if normalized:
            return self._normalized.sample()
        return self._unnormalized.sample()

    def _shape_check(self, val, func):
        low = self._unnormalized.low
        if hasattr(val, '__len__') and len(val) != len(low):
            raise TypeError(f'Unable to {func} value of length {len(val)}, expected {len(low)}')
        elif isinstance(val, (int, float)) and len(low) != 1:
            raise TypeError(f'Unable to {func} scalar value, expected array-like of shape {len(low)}')

    @abstractmethod
    def normalize(self, val):
        pass

    @abstractmethod
    def denormalize(self, val):
        pass

    @property
    def normalized(self):
        return self._normalized

    @property
    def unnormalized(self):
        return self._unnormalized

    def __getitem__(self, item):
        if item == 'normalized':
            return self._normalized
        elif item == 'unnormalized':
            return self._unnormalized

        raise KeyError(item)

    def clip(self, val, *, low=None, high=None,  space=None, normalized=None):
        """
        Clip a value into a lower and upper bound.

        The lower and upper bound are defined by the keyword argument ``low``, ``high``, ``normalized``, and ``space``.

        * If ``low`` and ``high`` are passed, ``val`` is clipped to the range ``[low, high]``.

        * Otherwise, if ``space`` is passed, ``val is clipped to the range ``[space.low, space.high]``.

        * Otherwise, if normalized is passed:

            * If ``normalized == True``, ``val is clipped to the range
            ``[space.normalized.low, space.normalized.high]``

            If ``normalized == False``, ``val is clipped to the range
            ``[space.unnormalized.low, space.unnormalized.high]``

        Parameters
        ----------
        val : np.ndarray or dict[str, list[np.ndarray]]
            Value to clip. Numpy array if ``isinstance(self, :class:`.ModuleSpace`)`` and a
            dict if ``isinstance(self, :class:`.MicrogridSpace`).``

        low : np.ndarray,  dict[str, list[np.ndarray]] or None, default None
            Value to define lower bound or None.

        high : np.ndarray,  dict[str, list[np.ndarray]] or None, default None
            Value to define upper bound or None.

        space : gym.Space or None, default None
            Space from which to select lower and upper bounds.

        normalized : bool or None, default None
            Whether to select bounds from ``self.normalized`` or ``self.unnormalized``.

        Returns
        -------
        clipped_val : np.ndarray or dict[str, list[np.ndarray]]
            Value clipped to lower and upper bounds.

        """
        if low is not None and high is not None:
            space = self._construct_space(low, high)
            return self.inner_clip(val, space)
        elif low is None and high is None:
            pass
        else:
            raise ValueError("One of 'low' or 'high' is None. Both should be or neither should be.")

        if space is None:
            if normalized is None:
                raise ValueError("No values to define low and high were passed!"
                                 "Must pass 'low' and 'high', or 'normalized', or 'space'.")
            elif normalized:
                space = self._normalized
            else:
                space = self._unnormalized

        return self.inner_clip(val, space)

    def _construct_space(self, low, high):
        # TODO (ahalev) this is not a good implementation
        Space = namedtuple('Space', ['low', 'high'])
        try:
            return {k: [Space(low=np.array(_v), high=np.array(high[k][j])) for j, _v in enumerate(v)] for k, v in low.items()}
        except AttributeError:
            return Space(low=np.array(low), high=np.array(high))

    @staticmethod
    def inner_clip(val, space):
        if (space.low > space.high).any():
            raise ValueError("Components of 'low' are greater than corresponding components in 'high'.")

        return np.clip(val, space.low, space.high)

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self._unnormalized).replace("Box", "")})'

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return self.unnormalized == other.unnormalized and self.normalized == other.normalized


class ModuleSpace(_PymgridSpace):
    _unnormalized: Box
    _normalized: Box
    _spread: np.ndarray

    def __init__(self,
                 unnormalized_low,
                 unnormalized_high,
                 normalized_bounds=(0, 1),
                 clip_vals=True,
                 shape=None,
                 dtype=np.float64,
                 seed=None,
                 verbose=False):

        self.clip_vals = clip_vals
        self.verbose = verbose

        low = np.float64(unnormalized_low) if np.isscalar(unnormalized_low) else unnormalized_low.astype(np.float64)
        high = np.float64(unnormalized_high) if np.isscalar(unnormalized_high) else unnormalized_high.astype(np.float64)

        self._unnormalized = Box(low=low,
                                 high=high,
                                 shape=shape,
                                 dtype=dtype)

        self._normalized = Box(low=normalized_bounds[0],
                               high=normalized_bounds[1],
                               shape=self._unnormalized.shape,
                               dtype=dtype)

        try:
            super().__init__(shape=self._unnormalized.shape, dtype=self._unnormalized.dtype, seed=seed)
        except TypeError:
            super().__init__(shape=self._unnormalized.shape, dtype=self._unnormalized.dtype)
            import gym
            warnings.warn(f"gym.Space does not accept argument 'seed' in version {gym.__version__}; this argument will "
                          f"be ignored. Upgrade your gym version with 'pip install -U gym' to use this functionality.")

        self._unnorm_spread = self._unnormalized.high - self._unnormalized.low
        self._unnorm_spread[self._unnorm_spread == 0] = 1

        self._norm_spread = self._normalized.high - self._normalized.low
        self._norm_spread[self._norm_spread == 0] = 1

    def normalize(self, val):
        un_low, un_high = self._unnormalized.low, self._unnormalized.high

        self._shape_check(val, 'normalize')
        val = self._bounds_check(val, un_low, un_high)

        normalized = self._normalized.low + (self._norm_spread / self._unnorm_spread) * (val - un_low)

        try:
            if not isinstance(val, np.ndarray):
                return normalized.item()
            return normalized
        except (AttributeError, ValueError):
            return normalized

    def denormalize(self, val):
        norm_low, norm_high = self._normalized.low, self._normalized.high

        self._shape_check(val, 'denormalize')
        val = self._bounds_check(val, norm_low, norm_high)

        denormalized = self._unnormalized.low + (self._unnorm_spread / self._norm_spread) * (val - norm_low)

        try:
            return denormalized.item()
        except (AttributeError, ValueError):
            return denormalized

    def _bounds_check(self, val, low, high):
        clipped = np.clip(val, low, high)

        if self.verbose or not self.clip_vals and (clipped != val).any():
            warnings.warn(f'Value {val} resides out of expected bounds of value to be normalized: [{low}, {high}].')

        if self.clip_vals:
            return clipped

        return val


class MicrogridSpace(_PymgridSpace):
    def __init__(self, unnormalized, normalized=None, seed=None):

        if normalized is None:
            if not isinstance(unnormalized, ModuleSpace):
                raise TypeError('Must pass both normalized and unnormalized or unnormalized must be ModuleSpace'
                                'with these aattributes.')

            normalized, unnormalized = unnormalized.normalized, unnormalized.unnormalized

        self._unnormalized = _PymgridDict(unnormalized)
        self._normalized = _PymgridDict(normalized)

        try:
            super().__init__(shape=None, seed=seed)

        except TypeError:
            super().__init__(shape=None)
            import gym
            warnings.warn(f"gym.Space does not accept argument 'seed' in version {gym.__version__}; this argument will "
                          f"be ignored. Upgrade your gym version with 'pip install -U gym' to use this functionality.")

        self._unnorm_spread = self._get_spread(normalized=False)
        self._norm_spread = self._get_spread(normalized=True)
        self._norm_over_unnorm = self.dict_op(self._norm_spread, self._unnorm_spread, operator.truediv)

    def _get_spread(self, normalized):
        if normalized:
            low, high = self._normalized.low, self._normalized.high
        else:
            low, high = self._unnormalized.low, self._unnormalized.high

        spread = {}

        for k, high_list in high.items():
            low_list = low[k]
            s = [h-l for h, l in zip(high_list, low_list)]
            for s_val in s:
                s_val[s_val == 0] = 1

            spread[k] = s

        return spread

    def normalize(self, val):
        self._shape_check(val, 'normalize')

        val_minus_low = self.dict_op(val, self._unnormalized.low, operator.sub)
        times_spread_ratio = self.dict_op(val_minus_low, self._norm_over_unnorm, operator.mul)
        plus_normalized_low = self.dict_op(times_spread_ratio, self._normalized.low, operator.add)

        return plus_normalized_low

    def denormalize(self, val):
        self._shape_check(val, 'denormalize')

        val_minus_low = self.dict_op(val, self._normalized.low, operator.sub)
        times_spread_ratio = self.dict_op(val_minus_low, self._norm_over_unnorm, operator.truediv)
        plus_unnormalized_low = self.dict_op(times_spread_ratio, self._unnormalized.low, operator.add)

        return plus_unnormalized_low

    @staticmethod
    def inner_clip(val, space):
        def op(v, s):
            return _PymgridSpace.inner_clip(v, s)
        return MicrogridSpace.dict_op(val, space, op)

    @staticmethod
    def dict_op(first, second, op):
        out = {}
        for k, first_list in first.items():
            second_list = second[k]
            out[k] = [op(f, s) for f, s in zip(first_list, second_list)]
        return out

    @classmethod
    def from_module_spaces(cls, module_spaces, act_or_obs):
        normalized = extract_builtins(module_spaces, act_or_obs, normalized=True)
        unnormalized = extract_builtins(module_spaces, act_or_obs, normalized=False)
        return cls(unnormalized, normalized)
