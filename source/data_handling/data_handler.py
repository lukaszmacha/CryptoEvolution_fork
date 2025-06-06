# data_handling/data_handler.py

import pandas as pd
from ..utils import Granularity
from source.coinbase import CoinBaseHandler, YahooFinanceHandler

class DataHandler():
    """
    Responsible for data handling. Including data collection and preparation.
    """

    def __init__(self, list_of_indicators_to_apply: list = []) -> None:
        """
        Class constructor.

        Parameters:
            list_of_indicators_to_apply (list): List of indicators further to apply.
        """

        self.indicators = list_of_indicators_to_apply
        self.coinbase = CoinBaseHandler()
        self.yahoo_finance = YahooFinanceHandler()

    async def prepare_data(self, trading_pair: str, start_date: str, end_date: str, granularity: Granularity) -> pd.DataFrame:
        """
        Collects data from coinbase API and extends it with assigned list of indicators.

        Parameters:
            trading_pair (str): String representing unique trainding pair symbol.
            start_date (str): String representing date that collected data should start from.
            end_date (str): String representing date that collected data should finish at.
            granularity (Granularity): Enum specifying resolution of collected data - e.g. each
                15 minutes or 1 hour or 6 hours is treated separately

        Raises:
            RuntimeError: If given traiding pair symbol is not recognized.

        Returns:
            (pd.DataFrame): Collected data extended with given indicators.
        """

        possible_traiding_pairs_cb = await self.coinbase.get_possible_pairs()
        possible_traiding_pairs_yf = await self.yahoo_finance.get_possible_pairs(asset_type='all')
        if trading_pair not in possible_traiding_pairs_cb.index and trading_pair not in possible_traiding_pairs_yf.index:
            raise RuntimeError('Traiding pair not recognized!')

        if trading_pair in possible_traiding_pairs_yf.index:
            data = await self.yahoo_finance.get_candles_for(trading_pair, start_date, end_date, granularity)
        else:
            data = await self.coinbase.get_candles_for(trading_pair, start_date, end_date, granularity)
        if self.indicators:
            indicators_data = []
            for indicator in self.indicators:
                indicators_data.append(indicator.calculate(data))
            data = pd.concat([data] + indicators_data, axis=1)

        return data
