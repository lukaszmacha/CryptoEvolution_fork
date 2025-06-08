# indicators/exponential_moving_average_indicator.py

from .indicator_base import *

class ExponentialMovingAverageIndicatorHandler(IndicatorHandlerBase):
    """
    Implements Exponential Moving Average (EMA) indicator.
    EMA places greater weight on recent data points compared to simple moving averages,
    making it more responsive to new information.
    """

    def __init__(self, window_size: int = 20) -> None:
        """
        Class constructor.

        Parameters:
            window_size (int): Period of days to calculate the EMA over.
        """
        self.window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the EMA indicator for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated EMA values.
        """
        ema_df = pd.DataFrame(index=data.index)
        ema_df[f'EMA_{self.window_size}'] = data['close'].ewm(span=self.window_size, adjust=False).mean()

        print(f"EMA NaN Count: {ema_df.notna().sum()}")
        return ema_df