# tests/test_indicators.py  

import pandas as pd
from source.indicators import VolumeProfileIndicatorHandler, StochasticOscillatorIndicatorHandler, DonchainChannelsIndicatorHandler, MovingVolumeProfileIndicatorHandler

INPUT_DATA = pd.DataFrame(data={
    'low': [20000.0, 20500.0, 20100.0, 20100.0, 20000.0],
    'high': [20900.0, 20900.0, 21000.0, 20900.0, 21700.0],
    'open': [20050.0, 20600.0, 20400.0, 20800.0, 20200.0],
    'close': [20600.0, 20400.0, 20800.0, 20200.0, 20900.0],
    'volume': [1000.0, 1200.0, 1100.0, 1300.0, 900.0]
})

def test_volume_profile_indicator():
    """
    Tests the VolumeProfileIndicatorHandler.

    Verifies that the VolumeProfileIndicatorHandler calculates the volume profile
    for given input data with a specified number of steps.

    Asserts:
        The result DataFrame matches the expected DataFrame.
    """

    expected = pd.DataFrame(data={
        'price': [19550.0, 20400.0, 21250.0],
        'volume': [2000.0, 3200.0, 300.0]
    })

    indicator = VolumeProfileIndicatorHandler(number_of_steps = 3)
    result = indicator.calculate(data=INPUT_DATA)
    pd.testing.assert_frame_equal(result, expected)

def test_stochastic_oscillator_indicator():
    """
    Tests the StochasticOscillatorIndicatorHandler.

    Verifies that the StochasticOscillatorIndicatorHandler calculates the K% and D%
    values for given input data with specified window size and D period.

    Expected DataFrame:
        K%: Calculated as (close - low) / (high - low) * 100 for each row.
        D%: Calculated as the rolling mean of K% with the specified D period.

    Asserts:
        The result DataFrame matches the expected DataFrame.
    """

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
        
    indicator = StochasticOscillatorIndicatorHandler(window_size = 3, d_period = 2)
    result = indicator.calculate(data=INPUT_DATA)
    pd.testing.assert_frame_equal(result, expected)

def test_donchain_channels_indicator():
    """
    Tests the DonchainChannelsIndicatorHandler.

    Verifies that the DonchainChannelsIndicatorHandler calculates the upper,
    lower, and middle Donchian channels for given input data with a specified window size.

    Asserts:
        The result DataFrame matches the expected DataFrame.
    """

    expected = pd.DataFrame(data={
        'upper_channel': [20900.0, 20900.0, 21000.0, 21000.0, 21700.0],
        'lower_channel': [20000.0, 20000.0, 20000.0, 20100.0, 20000.0],
        'middle_channel': [20450.0, 20450.0, 20500.0, 20550.0, 20850.0]
    })
        
    indicator = DonchainChannelsIndicatorHandler(window_size = 3)
    result = indicator.calculate(data=INPUT_DATA)
    pd.testing.assert_frame_equal(result, expected)

def test_moving_biggest_volume_price_indicator():
    """
    Test the MovingVolumeProfileIndicatorHandler.

    This test verifies that the MovingVolumeProfileIndicatorHandler calculates the moving
    volume profile for given input data with a specified window size and number of steps.

    Asserts:
        The result DataFrame matches the expected DataFrame.
    """

    expected = pd.DataFrame(data={
        'moving_volume_profile': [200.0, 800.0, 1070.0, 1395.0, 1580.0]
    })
        
    indicator = MovingVolumeProfileIndicatorHandler(window_size = 3, number_of_steps = 5)
    result = indicator.calculate(data=INPUT_DATA)
    pd.testing.assert_frame_equal(result, expected)