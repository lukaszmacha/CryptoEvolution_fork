# indicators/volume_profile_indicator.py

from .indicator_base import *
from collections import defaultdict

class VolumeProfileIndicator(IndicatorBase):

    def calculate(self, data: pd.DataFrame, number_of_steps: int = 40) -> pd.DataFrame:
        volume_profile = defaultdict(float)
        data_min = data['low'].min()
        data_max = data['high'].max()
        step = (data_max - data_min) / (number_of_steps - 1)

        for _, row in data.iterrows():
            equalized_low = row['low'] // step * step
            equalized_high = row['high'] // step * step
            price_range = np.linspace(equalized_low, equalized_high, num = int(round((equalized_high - equalized_low) / step + 1, 0)))
            price_range = np.round(price_range, 6 - int(np.floor(np.log10(data_min))))
            volume_per_step = row['volume'] / len(price_range)

            for price in price_range:
                volume_profile[price] += volume_per_step

        profile_df = pd.DataFrame(list(volume_profile.items()), columns=['price', 'volume'])
        profile_df.sort_values(by='price', inplace=True)
        return profile_df