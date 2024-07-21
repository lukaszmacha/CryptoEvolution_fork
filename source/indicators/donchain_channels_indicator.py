# indicators/donchain_channels_indicator.py 

from .indicator_base import *

class DonchainChannelsIndicatorHandler(IndicatorHandlerBase):

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        donchian_df = pd.DataFrame(index = data.index)
        donchian_df['upper_channel'] = data['high'].rolling(window = self.window_size, min_periods = 1).max()
        donchian_df['lower_channel'] = data['low'].rolling(window = self.window_size, min_periods = 1).min()
        donchian_df['middle_channel'] = (donchian_df['upper_channel'] + donchian_df['lower_channel']) / 2
        
        return donchian_df