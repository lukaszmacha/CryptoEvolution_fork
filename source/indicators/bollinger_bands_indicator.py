# indicators/bollinger_bands_indicator.py

from .indicator_base import *

class BollingerBandsIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Bollinger Bands indicator.
    Bollinger Bands consist of a middle band (SMA) with upper and lower bands
    at standard deviation levels, helping identify overbought and oversold conditions.
    """

    def __init__(self, window_size: int = 20, num_std_dev: float = 2.0) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Period to calculate the bands over.
            num_std_dev (float): Number of standard deviations for upper and lower bands.
        """
        self.window_size = window_size
        self.num_std_dev = num_std_dev

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Bollinger Bands indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated Bollinger Bands values.
        """
        bollinger_df = pd.DataFrame(index=data.index)

        # Calculate middle band (SMA)
        bollinger_df['BB_middle'] = data['close'].rolling(window=self.window_size, min_periods=1).mean()

        # Calculate standard deviation
        rolling_std = data['close'].rolling(window=self.window_size, min_periods=1).std()

        # Calculate upper and lower bands
        bollinger_df['BB_upper'] = bollinger_df['BB_middle'] + (self.num_std_dev * rolling_std)
        bollinger_df['BB_lower'] = bollinger_df['BB_middle'] - (self.num_std_dev * rolling_std)

        # Calculate bandwidth and %B
        bollinger_df['BB_bandwidth'] = (bollinger_df['BB_upper'] - bollinger_df['BB_lower']) / bollinger_df['BB_middle']
        bollinger_df['BB_percent_b'] = (data['close'] - bollinger_df['BB_lower']) / (bollinger_df['BB_upper'] - bollinger_df['BB_lower'])


        print(f"Bollinger Bands NaN Count: {bollinger_df.replace(np.nan, 0).notna().sum()}")
        bollinger_df = bollinger_df.fillna(0)
        return bollinger_df