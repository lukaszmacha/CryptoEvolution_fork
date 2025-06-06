# agent/strategies/classification_learning_strategy_handler.py

# global imports
from typing import Any
from tensorflow.keras.callbacks import Callback
import logging
import numpy as np

# local imports
from source.agent import LearningStrategyHandlerBase
from source.agent import AgentBase
from source.agent import ClassificationLearningAgent
from source.environment import TradingEnvironment
from source.model import BluePrintBase

class ClassificationLearningStrategyHandler(LearningStrategyHandlerBase):
    """"""

    # global constants
    PLOTTING_KEY: str = 'classification_learning'

    def create_agent(self, model_blue_print: BluePrintBase,
                     trading_environment: TradingEnvironment) -> AgentBase:
        """"""

        windows_size = trading_environment.get_trading_consts().WINDOW_SIZE
        spatial_data_shape = trading_environment.get_environment_spatial_data_dimension()
        market_data_shape = (spatial_data_shape[1] * windows_size, )

        number_of_classes = len(trading_environment.get_trading_consts().OUTPUT_CLASSES)
        model_adapter = model_blue_print.instantiate_model(market_data_shape, number_of_classes,
                                                           spatial_data_shape)
        return ClassificationLearningAgent(model_adapter)

    def fit(self, agent: ClassificationLearningAgent, environment: TradingEnvironment,
            nr_of_steps: int, nr_of_episodes: int, callbacks: list[Callback]) -> tuple[list[str], list[dict[str, Any]]]:
        """"""

        if not isinstance(agent, ClassificationLearningAgent):
            raise TypeError("Agent must be an instance of ClassificationLearningAgent.")

        input_data, output_data = environment.get_labeled_data()
        steps_per_epoch = nr_of_steps // nr_of_episodes
        batch_size = len(input_data) // steps_per_epoch
        if batch_size <= 0:
            logging.warning("Batch size is zero or negative, using value of 1 instead.")
            batch_size = 1

        env_length = environment.get_environment_length()
        currency_prices = environment.get_data_for_iteration(['close'], 0, env_length - 1)
        currency_prices = (np.array(currency_prices) / currency_prices[0]).tolist()

        return [ClassificationLearningStrategyHandler.PLOTTING_KEY], \
            [{"history": agent.classification_fit(input_data, output_data, batch_size = batch_size,
                                      epochs = nr_of_episodes, callbacks = callbacks),
              "currency_prices": currency_prices}]
