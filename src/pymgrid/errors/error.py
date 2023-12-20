class PymgridError(Exception):
    """
    Base pymgrid exception.
    """


class DeprecatedError(PymgridError):
    """
    Code deprecation error.
    """
