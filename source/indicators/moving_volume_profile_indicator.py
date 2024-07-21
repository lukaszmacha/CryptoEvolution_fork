# indicators/moving_volume_profile_indicator.py

from .volume_profile_indicator import *

class MovingVolumeProfileIndicatorHandler(VolumeProfileIndicatorHandler):

    def __init__(self, window_size: int = 14, number_of_steps: int = 40) -> None:
        super().__init__(number_of_steps)
        self.window_size = window_size

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        moving_volume_price_df = pd.DataFrame(index = data.index)

        for i, (index, row) in enumerate(data.iterrows()):
            volume_profiles_data = super().calculate(data[max(0, i - self.window_size) : i + 1])
            current_average_price = (row['low'] + row['high']) / 2
            volume_profiles_connected_to_lower_prices = volume_profiles_data['price'] <= current_average_price
            moving_volume_price_df.loc[index, 'moving_volume_profile'] = volume_profiles_data[volume_profiles_connected_to_lower_prices]['volume'].iloc[-1]

        return moving_volume_price_df