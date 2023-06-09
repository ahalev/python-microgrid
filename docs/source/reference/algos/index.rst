.. _api.control:

Control Algorithms
==================

.. currentmodule:: pymgrid.algos

Control algorithms built into pymgrid, as well as references for external algorithms that can be deployed

Rule Based Control
------------------

Heuristic Algorithm that deploys modules via a priority list.

.. autosummary::
    :toctree: ../api/algos/

    RuleBasedControl


Model Predictive Control
------------------------

Algorithm that depends on a future forecast as well as a model of state transitions to determine optimal controls.


.. autosummary::
    :toctree: ../api/algos/

    ModelPredictiveControl


Reinforcement Learning
----------------------

Algorithms that treat a microgrid as a Markov process, and train a black-box policy by repeated interactions with
the environment. See :doc:`here <../../examples/rl-example>` for an example of using
reinforcement learning to train such an algorithm.



..
   HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
   Copied from pandas docs.

   .. currentmodule:: pymgrid.algos.priority_list

   .. autosummary::
      :toctree: ../api/algos/priority_list/
      PriorityListElement