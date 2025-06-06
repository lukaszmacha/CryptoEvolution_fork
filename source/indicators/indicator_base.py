# indicators/indicator_base.py

import pandas as pd
import numpy as np

class IndicatorHandlerBase():
    """
    Base class for indicators. Enforces certain functions to be implemented
    in derivative classes.
    """

    def __init__(self, *args) -> None:
        """
        Class constructor. Parameters are specified in derivative classes.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates indicator values for given data.

        Parameters:
            data (pd.DataFrame): Data frame with input data.

        Returns:
            (pd.DataFrame): Output data with calculated values for certain indicator.
        """

        raise NotImplementedError("Subclasses must implement this method.")
