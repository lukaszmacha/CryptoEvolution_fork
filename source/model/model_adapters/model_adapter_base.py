# model/model_adapters/model_adapter_base.py

# global imports
from abc import ABC, abstractmethod
from typing import Any, Callable

class ModelAdapterBase(ABC):
    """"""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """"""

        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """"""

        pass

    @abstractmethod
    def print_summary(self, print_function: Callable = print) -> None:
        """"""

        pass

    @abstractmethod
    def fit(self, input_data: Any, output_data: Any, **kwargs) -> Any:
        """"""

        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """"""

        pass

    @abstractmethod
    def get_model(self) -> Any:
        """"""

        pass