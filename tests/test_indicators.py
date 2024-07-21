# tests/test_indicators.py  

import pandas as pd
from source.indicators import VolumeProfileIndicator, StochasticOscillatorIndicator, DonchainChannelsIndicator

INPUT_DATA = pd.DataFrame(data={
    'low': [20000, 20500, 20100, 20100, 20000],
    'high': [20900, 20900, 21000, 20900, 21700],
    'open': [20050, 20600, 20400, 20800, 20200],
    'close': [20600, 20400, 20800, 20200, 20900],
    'volume': [1000, 1200, 1100, 1300, 900]
})

def test_volume_profile_indicator():

    expected = pd.DataFrame(data={
        'price': [19550.0, 20400.0, 21250.0],
        'volume': [2000.0, 3200.0, 300.0]
    })

    indicator = VolumeProfileIndicator()
    result = indicator.calculate(data=INPUT_DATA, number_of_steps = 3)
    pd.testing.assert_frame_equal(result, expected)

def test_stochastic_oscillator_indicator():

    expected = pd.DataFrame(data={
        'K%': [6.0/9 * 100 , 4.0/9 * 100, 8.0/10 * 100, 1.0/9 * 100, 9.0/17 * 100],
        'D%': [
                6.0/9 * 100,
                (6.0/9 + 4.0/9) / 2 * 100,
                (4.0/9 + 8.0/10) / 2 * 100,
                (8.0/10 + 1.0/9) / 2 * 100,
                (1.0/9 + 9.0/17) / 2 * 100
        ]
    })
        
    indicator = StochasticOscillatorIndicator()
    result = indicator.calculate(data=INPUT_DATA, window_size = 3, d_period = 2)
    pd.testing.assert_frame_equal(result, expected)

def test_donchain_channels_indicator():

    expected = pd.DataFrame(data={
        'upper_channel': [20900.0, 20900.0, 21000.0, 21000.0, 21700.0],
        'lower_channel': [20000.0, 20000.0, 20000.0, 20100.0, 20000.0],
        'middle_channel': [20450.0, 20450.0, 20500.0, 20550.0, 20850.0]
    })
        
    indicator = DonchainChannelsIndicator()
    result = indicator.calculate(data=INPUT_DATA, window_size= 3)
    pd.testing.assert_frame_equal(result, expected)