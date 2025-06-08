# indicators/macd_indicator.py

from .indicator_base import *

class MACDIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Moving Average Convergence Divergence (MACD) indicator.
    MACD shows the relationship between two moving averages of a security's price.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> None:
        """
        Class constructor.

        Parameters:
            fast_period (int): Period for the fast EMA calculation.
            slow_period (int): Period for the slow EMA calculation.
            signal_period (int): Period for the signal line calculation.
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates MACD indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated MACD values.
        """
        macd_df = pd.DataFrame(index=data.index)

        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_df['MACD_line'] = fast_ema - slow_ema

        # Calculate signal line
        macd_df['MACD_signal'] = macd_df['MACD_line'].ewm(span=self.signal_period, adjust=False).mean()

        # Calculate MACD histogram
        macd_df['MACD_histogram'] = macd_df['MACD_line'] - macd_df['MACD_signal']

        print(f"MACD NaN Count: {macd_df.notna().sum()}")
        return macd_df