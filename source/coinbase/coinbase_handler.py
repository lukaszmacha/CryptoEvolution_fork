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

    def __convert_date_to_timestamp(self, date_str, date_format="%Y-%m-%d %H:%M:%S", target_timezone=pytz.UTC):
        return int(datetime.strptime(date_str, date_format).replace(tzinfo=target_timezone).timestamp())

    async def __send_request_to_coinbase(self, session, url, pid):
        try:
            async with session.get(url) as response:
                 data = await response.json()
                 if 'message' in data and data['message'] == 'Public rate limit exceeded':
                     raise ValueError("Exceeded public rate! Retrying in 30s...")
                 return data
        except:
            await asyncio.sleep(1)
            return await self.__send_request_to_coinbase(session, url, pid)

    async def get_candles_for(self, trading_pair, start_date, end_date, granularity):
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
    
    async def get_possible_pairs(self):
        async with aiohttp.ClientSession() as session:
            url = f'https://api.pro.coinbase.com/products/'
            response = await asyncio.gather(self.__send_request_to_coinbase(session, url, 0))
            data = [[product['id'], product['base_currency'], product['quote_currency']] for product in response[0]]
            df = pd.DataFrame(sorted(data), columns=['id', 'base_currency', 'quote_currency'])
            df.set_index('id', inplace=True)
            return df