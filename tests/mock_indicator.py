# tests/mock_indicator.py

from source.indicators.indicator_base import *

class MockIndicatorHandler(IndicatorHandlerBase):

    def __init__(self, mocking_fucntion) -> None:
        self.lambda_mocking_function = mocking_fucntion

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.lambda_mocking_function(data)