# agent/agent_base.py

# global imports
from typing import Optional, Callable
from source.model import ModelAdapterBase

# local imports

class AgentBase():
    """"""

    def __init__(self, model_adapter: ModelAdapterBase) -> None:
        """"""

        self._model_adapter: ModelAdapterBase = model_adapter

    def load_model(self, model_path: str) -> None:
        """"""

        self._model_adapter.load_model(model_path)

    def save_model(self, model_path: str) -> None:
        """"""

        self._model_adapter.save_model(model_path)

    def print_summary(self, print_function: Optional[Callable] = print) -> None:
        """"""

        self._model_adapter.print_summary(print_function)