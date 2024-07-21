# tests/test_data_handling.py

import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch
from source.data_handling import DataHandler
from source.utils import Granularity
from mock_indicator import MockIndicatorHandler

MOCKED_COINBASE_HANDLER_DATA = pd.DataFrame(data={
        'low': [8400.00, 8487.33, 8635.31],
        'high': [8752.34, 8973.45, 8927.45],
        'open': [8523.33, 8522.30, 8919.21],
        'close': [8522.31, 8915.00, 8757.84],
        'volume': [7353.139605, 10216.692545, 9152.706926]
    }, index = pd.DatetimeIndex(['2020-03-01', '2020-03-02', '2020-03-03'], name='time'))

@pytest.mark.asyncio
@patch('source.coinbase.CoinBaseHandler.get_candles_for', new_callable=AsyncMock)
async def test_prepare_data__no_indicators(mock_get_candles_for):

    mock_get_candles_for.return_value = MOCKED_COINBASE_HANDLER_DATA
    expected = MOCKED_COINBASE_HANDLER_DATA

    handler = DataHandler()
    result = await handler.prepare_data('BTC-USD', '2020-03-01 00:00:00', '2020-03-03 00:00:00', Granularity.ONE_DAY)
    pd.testing.assert_frame_equal(result, expected)

@pytest.mark.asyncio
@patch('source.coinbase.CoinBaseHandler.get_candles_for', new_callable=AsyncMock)
async def test_prepare_data__with_indicators(mock_get_candles_for):

    mock_get_candles_for.return_value = MOCKED_COINBASE_HANDLER_DATA

    mean_high_lambda = lambda data: data['high'].rolling(window = 2).mean().to_frame(name='mean_high')
    mean_high_mock_indicator = MockIndicatorHandler(mean_high_lambda)
    std_low_lambda = lambda data: data['low'].rolling(window = 2).std().to_frame(name='std_low')
    std_low_mock_indicator = MockIndicatorHandler(std_low_lambda)

    expected = pd.concat([MOCKED_COINBASE_HANDLER_DATA, mean_high_lambda(MOCKED_COINBASE_HANDLER_DATA), 
                          std_low_lambda(MOCKED_COINBASE_HANDLER_DATA)], axis=1)

    indicators = [mean_high_mock_indicator, std_low_mock_indicator]
    handler = DataHandler(indicators)
    result = await handler.prepare_data('BTC-USD', '2020-03-01 00:00:00', '2020-03-03 00:00:00', Granularity.ONE_DAY)
    pd.testing.assert_frame_equal(result, expected)