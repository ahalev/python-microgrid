from copy import deepcopy
from contextlib import contextmanager
from typing import Union

import pymgrid


@contextmanager
def dry_run(pymgrid_object: Union['pymgrid.Microgrid', 'pymgrid.modules.base.BaseMicrogridModule']):
    yield deepcopy(pymgrid_object)
