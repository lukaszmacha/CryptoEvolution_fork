# utils/learning_strategy_handler_from_string_converter.py

# global imports
from typing import Any, Type

# local imports
from source.utils import FromStringConverterBase
from source.agent import LearningStrategyHandlerBase, ReinforcementLearningStrategyHandler, \
    ClassificationLearningStrategyHandler

class LearningStrategyHandlerFromStringConverter(FromStringConverterBase):
    """"""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """"""

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[LearningStrategyHandlerBase]] = {
            'reinforcement_learning_strategy_handler': \
                ReinforcementLearningStrategyHandler,
            'classification_learning_strategy_handler': \
                ClassificationLearningStrategyHandler
        }