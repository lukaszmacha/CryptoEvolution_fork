# indicators/relative_strength_index_indicator.py

from .indicator_base import *

class RelativeStrengthIndexIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Relative Strength Index (RSI) indicator.
    RSI measures the speed and change of price movements, oscillating between 0 and 100.
    Values above 70 typically indicate overbought conditions, while values below 30 indicate oversold.
    """

    def __init__(self, window_size: int = 14) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Period to calculate the RSI over.
        """
        self.window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RSI indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated RSI values.
        """
        rsi_df = pd.DataFrame(index=data.index)

        # Calculate price changes
        delta = data['close'].diff()

        # Create gain (positive price changes) and loss (negative price changes) series
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=self.window_size, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.window_size, min_periods=1).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi_df['RSI'] = 100 - (100 / (1 + rs))

        # Handle division by zero
        rsi_df['RSI'] = rsi_df['RSI'].fillna(50)  # Neutral RSI when there's no data

        print(f"RSI NaN Count: {rsi_df.notna().sum()}")
        return rsi_df