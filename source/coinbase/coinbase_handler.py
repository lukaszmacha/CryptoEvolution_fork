# coinbase/coibase_handler.py
 
import aiohttp
import asyncio
from datetime import datetime
import pytz
import math
import pandas as pd
from ..utils import Granularity

MAX_NUMBER_OF_CANDLES_PER_REQUEST = 300

class CoinBaseHandler:
    """
    Responsible for handling request towards Coinbase API.
    """

    def __convert_date_to_timestamp(self, date_str: str, date_format: str = "%Y-%m-%d %H:%M:%S", target_timezone = pytz.UTC) -> int:
        """
        Converts date given by string into integer timestamp.

        Parameters:
            date_str (str): String representing certain date.
            date_format (str): String representing format that certain date is written in.
            target_timezone (Any): Timezone given by any type of value compatible with datetime library.

        Returns:
            (int): Timestamp converted from input date.
        """

        return int(datetime.strptime(date_str, date_format).replace(tzinfo=target_timezone).timestamp())

    async def __send_request_to_coinbase(self, session: aiohttp.ClientSession, url: str, pid: int) -> list:
        """
        Sends request towards Coinbase API and handles exceedance of public rates by repeating
        request after certain time.

        Parameters:
            session (aiohttp.ClientSession): Session used to send request with.
            url (str): URL address that certain request is sent towards.
            pid (int): Request indentification number.
        
        Raises:
            RuntimeError: If public rates are exceeded. Will try to handle that and reattempt 
                to sent request.

        Returns:
            (list): List of values returned by Coinbase API for certain request.
        """

        try:
            async with session.get(url) as response:
                 data = await response.json()
                 if 'message' in data and data['message'] == 'Public rate limit exceeded':
                     raise RuntimeError("Exceeded public rate! Retrying in 5s...")
                 return data
        except:
            await asyncio.sleep(5)
            return await self.__send_request_to_coinbase(session, url, pid)

    async def get_candles_for(self, trading_pair: str, start_date: str, end_date: str, granularity: Granularity) -> pd.DataFrame:
        """
        Collects data from Coinbase API given starting date, ending date, granularity and trainding pair. Dependent on amount of
        data segments to fetch, might take some time. Especially, if request exceeds public rates.

        Parameters:
            trading_pair (str): String representing unique trainding pair symbol.
            start_date (str): String representing date that collected data should start from.
            end_date (str): String representing date that collected data should finish at.
            granularity (Granularity): Enum specifying resolution of collected data - e.g. each 
                15 minutes or 1 hour or 6 hours is treated separately
        
        Raises:
            ValueError: If given granularity is not member if Granularity enum.

        Returns:
            (pd.DataFrame): Collected data frame.
        """

        if granularity not in Granularity:
            raise ValueError(f"{granularity} is not an value of Granularity enum!")
        
        start_timestamp = self.__convert_date_to_timestamp(start_date)
        end_timestamp = self.__convert_date_to_timestamp(end_date)
        granularity_seconds = granularity.value
        total_periods = (end_timestamp - start_timestamp) // granularity_seconds
        requests_needed = math.ceil(total_periods / MAX_NUMBER_OF_CANDLES_PER_REQUEST)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(requests_needed):
                start_period = start_timestamp + i * MAX_NUMBER_OF_CANDLES_PER_REQUEST * granularity_seconds
                end_period = min(start_period + MAX_NUMBER_OF_CANDLES_PER_REQUEST * granularity_seconds, end_timestamp)
                url = f'https://api.pro.coinbase.com/products/{trading_pair}/candles?start={start_period}&end={end_period}&granularity={granularity_seconds}'
                tasks.append(self.__send_request_to_coinbase(session, url, i))

            responses = await asyncio.gather(*tasks)
            candles = [item for sublist in responses if sublist for item in sublist]
            df = pd.DataFrame(candles, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.sort_values(by='time', inplace=True)
            return df
    
    async def get_possible_pairs(self) -> pd.DataFrame:
        """
        Collects data from Coinbase API reagrding all possible traiding pairs.

        Returns:
            (pd.DataFrame): Fetched possible traiding pairs inside data frame.
        """

        async with aiohttp.ClientSession() as session:
            url = f'https://api.pro.coinbase.com/products/'
            response = await asyncio.gather(self.__send_request_to_coinbase(session, url, 0))
            data = [[product['id'], product['base_currency'], product['quote_currency']] for product in response[0]]
            df = pd.DataFrame(sorted(data), columns=['id', 'base_currency', 'quote_currency'])
            df.set_index('id', inplace=True)
            return df