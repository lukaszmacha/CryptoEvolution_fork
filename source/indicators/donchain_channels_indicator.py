# indicators/donchain_channels_indicator.py 
from .indicator_base import *

class DonchainChannelsIndicator(IndicatorBase):

    def calculate(self, data: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
        donchian_df = pd.DataFrame(index=data.index)
        donchian_df['upper_channel'] = data['high'].rolling(window=window_size, min_periods=1).max()
        donchian_df['lower_channel'] = data['low'].rolling(window=window_size, min_periods=1).min()
        donchian_df['middle_channel'] = (donchian_df['upper_channel'] + donchian_df['lower_channel']) / 2
        
        return donchian_df