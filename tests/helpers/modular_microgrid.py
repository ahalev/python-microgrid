import numpy as np

from pymgrid import Microgrid

from pymgrid.modules import (
    BatteryModule,
    GensetModule,
    GridModule,
    LoadModule,
    RenewableModule
)


def get_modular_microgrid(remove_modules=(),
                          retain_only=None,
                          additional_modules=None,
                          add_unbalanced_module=True,
                          add_curtailment_module=True,
                          timeseries_length=100,
                          modules_only=False,
                          normalized_action_bounds=(0, 1)):

    modules = dict(
        genset=GensetModule(running_min_production=10,
                            running_max_production=50,
                            genset_cost=0.5,
                            normalized_action_bounds=normalized_action_bounds),

        battery=BatteryModule(min_capacity=0,
                              max_capacity=100,
                              max_charge=50,
                              max_discharge=50,
                              efficiency=1.0,
                              init_soc=0.5,
                              normalized_action_bounds=normalized_action_bounds),

        renewable=RenewableModule(time_series=50*np.ones(timeseries_length),
                                  normalized_action_bounds=normalized_action_bounds),

        load=LoadModule(time_series=60*np.ones(timeseries_length),
                        normalized_action_bounds=normalized_action_bounds),

        grid=GridModule(max_import=100,
                        max_export=0,
                        time_series=np.ones((timeseries_length, 3)),
                        normalized_action_bounds=normalized_action_bounds,
                        raise_errors=True)
        )

    if retain_only is not None:
        modules = {k: v for k, v in modules.items() if k in retain_only}
        if remove_modules:
            raise RuntimeError('Can pass either remove_modules or retain_only, but not both.')
    else:
        for module in remove_modules:
            try:
                modules.pop(module)
            except KeyError:
                raise NameError(f"Module {module} not one of default modules {list(modules.keys())}.")

    modules = list(modules.values())
    modules.extend(additional_modules if additional_modules else [])

    if modules_only:
        return modules

    return Microgrid(modules,
                     add_unbalanced_module=add_unbalanced_module,
                     add_curtailment_module=add_curtailment_module)
