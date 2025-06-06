# utils/testing_strategy_handler_from_string_converter.py

# global imports
from typing import Any, Type

# local imports
from source.utils import FromStringConverterBase
from source.agent import TestingStrategyHandlerBase, PerformanceTestingStrategyHandler, \
    ClassificationTestingStrategyHandler

class TestingStrategyHandlerFromStringConverter(FromStringConverterBase):
    """"""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """"""

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[TestingStrategyHandlerBase]] = {
            'performance_testing_strategy_handler': PerformanceTestingStrategyHandler,
            'classification_testing_strategy_handler': ClassificationTestingStrategyHandler
        }