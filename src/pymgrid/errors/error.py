class PymgridError(Exception):
    """
    Base pymgrid exception.
    """


class DeprecationError(PymgridError):
    """
    Code deprecation error.
    """
