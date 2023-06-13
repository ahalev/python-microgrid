import inspect
import yaml

from warnings import warn
from abc import abstractmethod


class BaseRewardShaper(yaml.YAMLObject):
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    @staticmethod
    def sum_module_val(info, module_name, module_attr):
        try:
            module_info = info[module_name]
            return sum([d[module_attr] for d in module_info])
        except KeyError:
            return 0.0

    @staticmethod
    def compute_net_load(step_info):
        return BaseRewardShaper.compute_total_load(step_info) - BaseRewardShaper.compute_total_renewable(step_info)

    @staticmethod
    def compute_total_load(step_info):
        try:
            load_info = step_info['load']
        except KeyError:
            raise NameError("Microgrid has no module with name 'load'")

        return sum(d['absorbed_energy'] for d in load_info)

    @staticmethod
    def compute_total_renewable(step_info):
        try:
            renewable_info = step_info.get('renewable', step_info['pv'])
        except KeyError:
            raise NameError("Microgrid has no module with name 'renewable' or 'pv'.")

        return sum(d['provided_energy'] for d in renewable_info)

    @abstractmethod
    def __call__(self, original_reward, step_info, cost_info):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def serializable_state_attributes(self):
        return []

    def serialize(self):
        return {attr: getattr(self, attr) for attr in self.serializable_state_attributes()}

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, data.serialize(), flow_style=cls.yaml_flow_style)

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        instance = cls.deserialize_instance(mapping)
        instance.set_state_attributes(mapping)
        return instance

    @classmethod
    def deserialize_instance(cls, param_dict):
        """
        Generate an instance of this module with the arguments in param_dict.

        Part of the ``load`` and ``yaml.safe_load`` methods. Should not be called directly.

        :meta private:

        Parameters
        ----------
        param_dict : dict
            Class arguments.

        Returns
        -------
        BaseMicrogridModule or child class of BaseMicrogridModule
            The module instance.

        """
        param_dict = param_dict.copy()
        cls_params = inspect.signature(cls).parameters

        cls_kwargs = {}
        missing_params, default_params = [], []

        for p_name, p_value in cls_params.items():
            try:
                cls_kwargs[p_name] = param_dict.pop(p_name)
            except KeyError:
                if p_value.default is p_value.empty:
                    missing_params.append(p_name)
                else:
                    cls_kwargs[p_name] = p_value.default
                    default_params.append(p_name)

        if len(default_params):
            warn(f'Missing parameter values {default_params} for {cls}. Using available default values.')

        if len(missing_params):
            raise KeyError(f"Missing parameter values {missing_params} for {cls} with no default values available.")

        return cls(**cls_kwargs)

    def set_state_attributes(self, attrs_dict):
        for attr in self.serializable_state_attributes():
            value = attrs_dict.get(attr, getattr(self, attr))
            setattr(self, attr, value)
