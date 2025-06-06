# agent/strategies/testing_strategy_handler_base.py

# global imports
from typing import Any
from abc import ABC, abstractmethod

# local imports
from source.environment import TradingEnvironment
from source.agent import AgentBase

class TestingStrategyHandlerBase(ABC):
    """"""

    @abstractmethod
    def evaluate(self, agent: AgentBase, environment: TradingEnvironment) -> \
        tuple[list[str], list[dict[str, Any]]]:
        """"""

        pass