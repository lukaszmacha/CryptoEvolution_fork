# agent/strategies/classification_testable.py

# global imports
import pandas as pd
from abc import ABC, abstractmethod

# local imports

class ClassificationTestable(ABC):
    """"""

    @abstractmethod
    def classify(self, data: pd.DataFrame) -> list[list[float]]:
        """"""

        pass