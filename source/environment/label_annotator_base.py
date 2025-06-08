# agent/label_annotator_base.py

# global imports
from abc import ABC, abstractmethod
import pandas as pd
from types import SimpleNamespace
import logging

# local imports

class LabelAnnotatorBase(ABC):
    """"""

    __CLOSE_PRICE_COLUMN_NAME: str = "close"

    def __init__(self) -> None:
        """"""

        self._output_classes = SimpleNamespace()

    @abstractmethod
    def _classify_trend(self, price_diff: float, volatility: float) -> int:
        """"""

        pass

    def annotate(self, data: pd.DataFrame) -> pd.Series:
        """"""

        current_prices = data[self.__CLOSE_PRICE_COLUMN_NAME]
        next_day_prices = data[self.__CLOSE_PRICE_COLUMN_NAME].shift(-1)
        price_diffs = (next_day_prices - current_prices) / current_prices
        price_diffs = price_diffs.dropna()
        price_diffs.reset_index(drop=True, inplace=True)
        volatilities = data["volatility"][:-1]

        logging.info(f"{volatilities.shape} {price_diffs.shape}")

        # Create a new Series with the same index as price_diffs
        result = pd.Series(index=price_diffs.index, dtype=int)

        # Fill the result with classified values
        for idx in price_diffs.index:
            if pd.notna(price_diffs[idx]) and pd.notna(volatilities[idx]):
                result[idx] = self._classify_trend(price_diffs[idx], volatilities[idx])

        logging.info(f"Label Annotator NaN Count: {result.isna().sum()}")
        logging.info(f"Label Annotator Class Count: {result.shape}")
        return result

    def get_output_classes(self) -> SimpleNamespace:
        """"""

        return self._output_classes