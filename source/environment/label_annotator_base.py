# agent/label_annotator_base.py

# global imports
from abc import ABC, abstractmethod
import pandas as pd
from types import SimpleNamespace

# local imports

class LabelAnnotatorBase(ABC):
    """"""

    __CLOSE_PRICE_COLUMN_NAME: str = "close"

    def __init__(self) -> None:
        """"""

        self._output_classes = SimpleNamespace()

    @abstractmethod
    def _classify_trend(self, price_diff: float) -> int:
        """"""

        pass

    def annotate(self, data: pd.DataFrame) -> pd.Series:
        """"""

        current_prices = data[self.__CLOSE_PRICE_COLUMN_NAME]
        next_day_prices = data[self.__CLOSE_PRICE_COLUMN_NAME].shift(-1)
        price_diffs = (next_day_prices - current_prices) / current_prices

        return price_diffs.apply(self._classify_trend)

    def get_output_classes(self) -> SimpleNamespace:
        """"""

        return self._output_classes