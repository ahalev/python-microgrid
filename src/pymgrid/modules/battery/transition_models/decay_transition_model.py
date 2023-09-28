from pymgrid.modules.battery.transition_models import BatteryTransitionModel


class DecayTransitionModel(BatteryTransitionModel):
    yaml_tag = u"!DecayTransitionModel"

    def __init__(self, decay_rate=0.999**(1/24)):
        """

        Parameters
        ----------
        decay_rate : float, default 0.99**(1/24)
            Amount to decay in one time step; should be in (0, 1]. If 1, no decay and equivalent to parent model.
            Default is equivalent to 1/10 of a percent decay in one day.

        """
        self.decay_rate = decay_rate
        self.initial_step = None

    def _current_efficiency(self, efficiency, current_step):
        return efficiency * (self.decay_rate ** (current_step-self.initial_step))

    def _update_step(self, current_step):
        if self.initial_step is None or current_step <= self.initial_step:
            self.initial_step = current_step

    def transition(self, external_energy_change, efficiency, current_step, **kwargs):
        self._update_step(current_step)
        current_efficiency = self._current_efficiency(efficiency, current_step)

        return super().transition(external_energy_change, efficiency=current_efficiency)


class DecayCycleTransitionModel(DecayTransitionModel):
    # https://en.wikipedia.org/wiki/Capacity_loss

    yaml_tag = u"!DecayCycleTransitionModel"

    def __init__(self, decay_rate=1-2.48e-4):
        super().__init__(None)
        self.decay_rate_per_cycle = decay_rate
        self.cycle_amount = None
        self.num_cycles = 0

    def _current_efficiency(self, efficiency, current_step):
        self.decay_rate = self.decay_rate_per_cycle ** self.num_cycles
        return super()._current_efficiency(efficiency, current_step)

    def _update_num_cycles(self, external_energy_change, max_capacity, min_capacity):
        if self.cycle_amount is None:
            self.cycle_amount = max_capacity - min_capacity

        if external_energy_change < 0:
            return

        self.num_cycles += external_energy_change / self.cycle_amount

    def transition(self, external_energy_change, efficiency, current_step, max_capacity, min_capacity, **kwargs):
        self._update_num_cycles(external_energy_change, max_capacity, min_capacity)

        return super().transition(external_energy_change, efficiency, current_step)
