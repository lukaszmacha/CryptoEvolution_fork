# agent/strategies/learning_strategy_handler_base.py

# global imports
from typing import Any
from tensorflow.keras.callbacks import Callback

# local imports
from source.environment import TradingEnvironment
from source.agent import AgentBase
from source.model import BluePrintBase

class LearningStrategyHandlerBase:
    """"""

    def create_agent(self, model_blue_print: BluePrintBase,
                     trading_environment: TradingEnvironment) -> AgentBase:
        """"""

        raise NotImplementedError("Subclasses must implement this method.")

    def fit(self, agent: AgentBase, nr_of_steps: int, nr_of_episodes: int,
            callbacks: list[Callback]) -> tuple[list[str], list[dict[str, Any]]]:
        """"""

        raise NotImplementedError("Subclasses must implement this method.")