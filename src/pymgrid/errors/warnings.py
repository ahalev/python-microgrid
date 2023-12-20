class PymgridWarning(Warning):
    """
    Base pymgrid warning.
    """


class PymgridDeprecationWarning(PymgridWarning, DeprecationWarning):
    """
    Pymgrid deprecation warning.
    """