.. _api.modules:

Modules
=======

.. currentmodule:: pymgrid.modules

The modules defined here are commonly found in microgrids.
Pass any combination of modules to :ref:`Microgrid <api.microgrid>` to define and run a microgrid.

Timeseries Modules
------------------

Modules that are temporal in nature.



.. autosummary::
    :toctree: ../api/modules/

    GridModule
    LoadModule
    RenewableModule

Non-temporal Modules
--------------------

Modules that do not depend on an underlying timeseries.

.. autosummary::
    :toctree: ../api/modules/

    BatteryModule
    GensetModule

Helper Module
--------------

A module that cleans up after all the other modules are deployed.

.. autosummary::
    :toctree: ../api/modules/

    UnbalancedEnergyModule


Module Functions
================

Battery Transition Models
-------------------------

Various battery transition models.

.. currentmodule:: pymgrid.modules.battery.transition_models

.. autosummary::
    :toctree: ../api/battery_transition_models/

    BatteryTransitionModel
    BiasedTransitionModel
    DecayTransitionModel
=======
----------------

.. toctree::
   :maxdepth: 1

   battery_transition_models/index
