# agent/simple_label_annotator.py

# global imports
import pandas as pd
from types import SimpleNamespace

# local imports
from source.environment import LabelAnnotatorBase

class SimpleLabelAnnotator(LabelAnnotatorBase):
    """"""

    def __init__(self, alpha: float = 0.55) -> None:
        """"""

        super().__init__()
        self._output_classes.UP_TREND = 0
        self._output_classes.DOWN_TREND = 1
        self._output_classes.NO_TREND = 2
        self.__alpha = alpha

    def _classify_trend(self, price_diff: float, volatility: float) -> int:
        """"""

        if price_diff > self.__alpha // 10:
            return self._output_classes.UP_TREND
        elif price_diff < -self.__alpha // 10:
            return self._output_classes.DOWN_TREND
        else:
            return self._output_classes.NO_TREND
