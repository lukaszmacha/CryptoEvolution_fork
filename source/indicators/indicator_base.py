# indicators/indicator_base.py

import pandas as pd
import numpy as np

class IndicatorHandlerBase():

    def __init__(self, *args) -> None:
        raise NotImplementedError
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        