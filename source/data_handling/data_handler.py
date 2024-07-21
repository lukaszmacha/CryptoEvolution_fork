# data_handling/data_handler.py

import pandas as pd
from ..utils import Granularity
from ..coinbase import CoinBaseHandler

class DataHandler():

    def __init__(self, list_of_indicators_to_apply: list = []) -> None:
        self.indicators = list_of_indicators_to_apply
        self.coinbase = CoinBaseHandler()

    async def prepare_data(self, trading_pair: str, start_date: str, end_date: str, granularity: Granularity) -> pd.DataFrame:
        data = await self.coinbase.get_candles_for(trading_pair, start_date, end_date, granularity)
        
        if self.indicators:
            indicators_data = []
            for indicator in self.indicators:
                indicators_data.append(indicator.calculate(data))
            data = pd.concat([data] + indicators_data, axis=1)

        return data



