# tests/test_coinbase.py

import pandas as pd
from source.coinbase import CoinBaseHandler
from source.utils import Granularity
import pytest

@pytest.mark.asyncio
async def test_get_candles_for():
    """
    Tests the get_candles_for method of CoinBaseHandler.

    Verifies that the get_candles_for method retrieves the correct candle data
    for a given trading pair and date range. The expected data is hardcoded for the 
    date range from '2020-03-01' to '2020-03-03' for the 'BTC-USD' pair with a granularity
    of one day.

    Asserts:
        The result DataFrame matches the expected DataFrame.
    """

    expected = pd.DataFrame(data={
        'low': [8400.00, 8487.33, 8635.31],
        'high': [8752.34, 8973.45, 8927.45],
        'open': [8523.33, 8522.30, 8919.21],
        'close': [8522.31, 8915.00, 8757.84],
        'volume': [7353.139605, 10216.692545, 9152.706926]
    }, index = pd.DatetimeIndex(['2020-03-01', '2020-03-02', '2020-03-03'], name='time'))      

    handler = CoinBaseHandler()
    result = await handler.get_candles_for('BTC-USD', '2020-03-01 00:00:00', '2020-03-03 00:00:00', Granularity.ONE_DAY)
    pd.testing.assert_frame_equal(result, expected)

@pytest.mark.asyncio
async def test_get_possible_pairs():
    """
    Tests the get_possible_pairs method of CoinBaseHandler.

    Verifies that the get_possible_pairs method retrieves the correct trading pairs
    available on the exchange. The expected data is hardcoded for the first three trading pairs.

    Asserts:
        The result DataFrame matches the expected DataFrame.
    """
        
    expected = pd.DataFrame(data={
        'base_currency': ['00','1INCH', '1INCH'],
        'quote_currency': ['USD', 'BTC', 'EUR']
    }, index=pd.Index(['00-USD', '1INCH-BTC', '1INCH-EUR'], name='id'))      

    handler = CoinBaseHandler()
    result = await handler.get_possible_pairs()
    pd.testing.assert_frame_equal(result[:3], expected)
