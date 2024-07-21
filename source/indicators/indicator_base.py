# indicators/indicator_base.py

import pandas as pd
import numpy as np

class IndicatorBase():

    def calculate(self, data: pd.DataFrame, *args) -> pd.DataFrame:
        raise NotImplementedError
        