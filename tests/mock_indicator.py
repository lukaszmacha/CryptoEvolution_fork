# tests/mock_indicator.py

from source.indicators.indicator_base import *
from typing import Callable

class MockIndicatorHandler(IndicatorHandlerBase):
    """
    Implements indicator that calculated values can be specified from outside of the class.
    This allows for creation of simple indicators to be mocked during testing.
    """

    def __init__(self, mocking_fucntion: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """
        Class constructor.

        Parameters:
            mocking_fucntion (Callable[[pd.DataFrame], pd.DataFrame]): Allows to specify
                functionality of the indicator from outside of the class.
        """

        self.lambda_mocking_function = mocking_fucntion

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates values for given data by applying mocking function.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated values.
        """

        return self.lambda_mocking_function(data)