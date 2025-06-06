# coinbase/yahoo_finance_handler.py

import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
from source.utils import Granularity

class YahooFinanceHandler:
    """
    Responsible for handling requests towards Yahoo Finance API.
    """

    def __map_granularity_to_yahoo(self, granularity: Granularity) -> str:
        """
        Maps internal Granularity enum to Yahoo Finance interval format.

        Parameters:
            granularity (Granularity): Granularity enum value

        Returns:
            (str): Yahoo Finance compatible interval string
        """
        # Yahoo Finance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        mapping = {
            Granularity.ONE_MINUTE: "1m",
            Granularity.FIVE_MINUTES: "5m",
            Granularity.FIFTEEN_MINUTES: "15m",
            Granularity.ONE_HOUR: "60m",
            Granularity.SIX_HOURS: "1h",  # No exact match, using 1h
            Granularity.ONE_DAY: "1d"
        }

        if granularity not in mapping:
            raise ValueError(f"Granularity {granularity} not supported by Yahoo Finance API")

        return mapping[granularity]

    async def get_possible_pairs(self, asset_type="stocks") -> pd.DataFrame:
        """
        Returns commonly used trading pairs on Yahoo Finance based on asset type.

        Parameters:
            asset_type (str): Type of assets to return. Options are:
                            "stocks" - Common stock tickers
                            "indices" - Major market indices
                            "all" - All available assets

        Returns:
            (pd.DataFrame): Common trading pairs inside data frame.
        """

        # Define common stock tickers
        stock_pairs = [
            ['AAPL', 'Apple Inc.', 'USD'],
            ['MSFT', 'Microsoft Corporation', 'USD'],
            ['GOOGL', 'Alphabet Inc.', 'USD'],
            ['AMZN', 'Amazon.com Inc.', 'USD'],
            ['TSLA', 'Tesla Inc.', 'USD'],
            ['META', 'Meta Platforms Inc.', 'USD'],
            ['NVDA', 'NVIDIA Corporation', 'USD'],
            ['JPM', 'JPMorgan Chase & Co.', 'USD'],
            ['V', 'Visa Inc.', 'USD'],
            ['WMT', 'Walmart Inc.', 'USD']
        ]

        # Define common indices
        indices_pairs = [
            ['^GSPC', 'S&P 500', 'USD'],
            ['^DJI', 'Dow Jones Industrial Average', 'USD'],
            ['^IXIC', 'NASDAQ Composite', 'USD'],
            ['^RUT', 'Russell 2000', 'USD'],
            ['^FTSE', 'FTSE 100', 'GBP'],
            ['^N225', 'Nikkei 225', 'JPY'],
            ['^HSI', 'Hang Seng Index', 'HKD'],
            ['^GDAXI', 'DAX', 'EUR'],
            ['^FCHI', 'CAC 40', 'EUR'],
            ['^STOXX50E', 'EURO STOXX 50', 'EUR']
        ]

        # Return appropriate pairs based on asset_type
        if asset_type.lower() == "stocks":
            pairs = stock_pairs
            columns = ['id', 'company_name', 'currency']
        elif asset_type.lower() == "indices":
            pairs = indices_pairs
            columns = ['id', 'index_name', 'currency']
        elif asset_type.lower() == "all":
            # Combine all pairs with appropriate identifiers
            stocks_tagged = [[p[0], 'stock', p[1], p[2]] for p in stock_pairs]
            indices_tagged = [[p[0], 'index', p[1], p[2]] for p in indices_pairs]
            pairs = stocks_tagged + indices_tagged
            columns = ['id', 'type', 'name', 'currency']
        else:
            raise ValueError(f"Unknown asset type: {asset_type}. Expected 'stocks', 'indices', or 'all'")

        df = pd.DataFrame(pairs, columns=columns)
        df.set_index('id', inplace=True)

        return df

    async def get_candles_for(self, symbol: str, start_date: str, end_date: str, granularity: Granularity) -> pd.DataFrame:
        """
        Collects data from Yahoo Finance API given starting date, ending date, granularity and symbol.
        Works for cryptocurrencies, stocks, and indices.

        Parameters:
            symbol (str): String representing unique symbol (e.g. "BTC-USD", "AAPL", "^GSPC").
            start_date (str): String representing date that collected data should start from.
            end_date (str): String representing date that collected data should finish at.
            granularity (Granularity): Enum specifying resolution of collected data.

        Raises:
            ValueError: If given granularity is not supported by Yahoo Finance.

        Returns:
            (pd.DataFrame): Collected data frame.
        """
        if granularity not in Granularity:
            raise ValueError(f"{granularity} is not a value of Granularity enum!")

        interval = self.__map_granularity_to_yahoo(granularity)

        # Run the Yahoo Finance download in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval
            )
        )

        # Rename columns to match Coinbase format
        df = df.rename(columns={
            'Low': 'low',
            'High': 'high',
            'Open': 'open',
            'Close': 'close',
            'Volume': 'volume'
        })

        df.columns = df.columns.get_level_values(0)
        return df
