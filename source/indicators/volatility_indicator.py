# indicators/volatility_indicator.py

from .indicator_base import *

class VolatilityIndicatorHandler(IndicatorHandlerBase):
    """
    Implements volatility measurement based on the standard deviation of past data points.

    This indicator calculates volatility using the standard deviation of percentage
    price changes over the past specified number of data points (default: 10).
    """

    def __init__(self, window_size: int = 10) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Number of past data points to consider for volatility calculation.
                               Default is 10 as specified in the study.
        """
        self.window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates volatility indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data (must contain 'close' column).

        Returns:
            (pd.DataFrame): Output data with calculated volatility values.
        """
        volatility_df = pd.DataFrame(index=data.index)

        # Calculate percentage changes of closing prices
        pct_changes = data['close'].pct_change()

        # Calculate rolling standard deviation of percentage changes
        volatility_df['volatility'] = pct_changes.rolling(window=self.window_size, min_periods=1).std()

        # Fill NaN values with 0 for the first data point
        volatility_df = volatility_df.fillna(0)

        print(f"Volatility NaN Count: {volatility_df.isna().sum()}")

        return volatility_df