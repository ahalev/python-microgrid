import unittest
import numpy as np
import pandas as pd


class TestCase(unittest.TestCase):
    def assertEqual(self, first, second, msg=None) -> None:
        try:
            super().assertEqual(first, second, msg=msg)
        except (ValueError, AssertionError):
            # array-like or pandas obj
                # convert pandas obj
            if isinstance(first, (pd.DataFrame, pd.Series)):
                first, second = first.values, second.values

            try:
                np.testing.assert_equal(first, second, err_msg=msg if msg else '')
            except AssertionError as e:
                try:
                    np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-10, err_msg=msg if msg else '')
                except TypeError:
                    raise e
