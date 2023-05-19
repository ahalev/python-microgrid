import numpy as np

from pymgrid.modules import RenewableModule

from tests.microgrid.modules.module_tests.timeseries_modules import (
    TestTimeseriesModuleForecasting,
    TestTimeseriesModuleNoForecasting,
    TestTimeSeriesModuleForecastingNegativeVals,
    TestTimeSeriesModuleNoForecastingNegativeVals
)


class TestRenewableModuleNoForecasting(TestTimeseriesModuleNoForecasting):
    __test__ = True
    action_space_dim = 0

    def get_module(self):
        return RenewableModule(self.module_time_series)

    def test_init_current_renewable(self):
        renewable_module = self.get_module()
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])

    def test_step(self):
        renewable_module = self.get_module()
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])

        obs, reward, done, info = renewable_module.step(np.array([]))
        obs = renewable_module.from_normalized(obs, obs=True)
        self.assertEqual(obs, self.time_series[1])
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["provided_energy"], self.time_series[0])

class TestRenewableModuleForecasting(TestTimeseriesModuleForecasting):
    __test__ = True
    action_space_dim = 0

    def get_module(self):
        return RenewableModule(self.module_time_series, forecaster="oracle", forecast_horizon=self.forecast_horizon)

    def test_step(self):
        renewable_module = self.get_module()
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])

        action = renewable_module.to_normalized(np.array([]), act=True)
        obs, reward, done, info = renewable_module.step(action)
        obs = renewable_module.from_normalized(obs, obs=True)
        self.assertEqual(obs, self.time_series[1:self.forecast_horizon+2])
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["provided_energy"], self.time_series[0])


class TestRenewableModuleForecastingNegativeVals(TestTimeSeriesModuleForecastingNegativeVals,
                                                 TestRenewableModuleForecasting):
    pass


class TestRenewableModuleNoForecastingNegativeVals(TestTimeSeriesModuleNoForecastingNegativeVals,
                                                   TestRenewableModuleNoForecasting):
    pass
