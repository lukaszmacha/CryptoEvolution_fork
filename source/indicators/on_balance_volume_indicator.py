# indicators/on_balance_volume_indicator.py

from .indicator_base import *

class OnBalanceVolumeIndicatorHandler(IndicatorHandlerBase):
    """
    Implements On-Balance Volume (OBV) indicator.
    OBV measures buying and selling pressure as a cumulative indicator,
    adding volume on up days and subtracting volume on down days.
    """

    def __init__(self) -> None:
        """
        Class constructor.
        """
        pass

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates OBV indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated OBV values.
        """
        obv_df = pd.DataFrame(index=data.index)

        # Calculate price changes
        close_change = data['close'].diff()

        # Initialize OBV with the first volume value
        obv = [0]

        # Calculate OBV values
        for i in range(1, len(data)):
            if close_change.iloc[i] > 0:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif close_change.iloc[i] < 0:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        obv_df['OBV'] = obv

        print(f"On-Balance Volume NaN Count: {obv_df.notna().sum()}")
        return obv_df