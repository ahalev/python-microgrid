import numpy as np

import yaml

from pymgrid.modules.base import BaseMicrogridModule


class RenewableCurtailmentModule(BaseMicrogridModule):
    def update(self, external_energy_change, as_source=False, as_sink=False):
        pass

    def _state_dict(self):
        pass

    @property
    def min_obs(self):
        pass

    @property
    def max_obs(self):
        pass

    @property
    def min_act(self):
        pass

    @property
    def max_act(self):
        pass