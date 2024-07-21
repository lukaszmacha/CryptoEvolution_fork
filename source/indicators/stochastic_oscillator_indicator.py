# indicators/stochastic_oscillator_indicator.py

from .indicator_base import *

class StochasticOscillatorIndicatorHandler(IndicatorHandlerBase):

    def __init__(self, window_size: int = 14, d_period: int = 3) -> None:
        self.window_size = window_size
        self.d_period = d_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        high_series = data['high']
        low_series = data['low']
        close_series = data['close']
        
        highest_high = high_series.rolling(window = self.window_size, min_periods = 1).max()
        lowest_low = low_series.rolling(window = self.window_size, min_periods = 1).min()
        
        stochastic_data_df = pd.DataFrame(index = data.index)
        stochastic_data_df['K%'] = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
        stochastic_data_df['D%'] = stochastic_data_df['K%'].rolling(window = self.d_period, min_periods = 1).mean()

        return stochastic_data_df