# training/training_handler.py

import logging
import io
from  tensorflow.keras.callbacks import Callback 
from typing import Optional

from .training_config import TrainingConfig
from ..environment.trading_environment import TradingEnvironment
from ..agent.agent_hanlder import AgentHandler

class TrainingHandler():
    """
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        """

        self.__environment: TradingEnvironment = config.instantiate_environment()
        self.__agent: AgentHandler = config.instantiate_agent()
        self.__nr_of_steps: int = config.nr_of_steps
        self.__repeat_test: int = config.nr_of_steps
        self.__steps_per_episode: int = int(config.nr_of_steps / config.nr_of_episodes)
        self.__generated_data: dict = {}
        self.__logs: io.StringIO = io.StringIO()

    def run_training(self, callbacks: list[Callback] = [], weights_load_path: Optional[str] = None,
                     weights_save_path: Optional[str] = None) -> None:
        """
        """

        logging.basicConfig(stream = self.__logs, level = logging.INFO)

        self.__generated_data['train'] = self.__agent.train_agent(self.__environment, 
                                                                  self.__nr_of_steps,
                                                                  self.__steps_per_episode, 
                                                                  callbacks, 
                                                                  weights_load_path, 
                                                                  weights_save_path)

        self.__generated_data['test'] = self.__agent.test_agent(self.__environment,
                                                                self.__repeat_test)

    def generate_raport(self) -> None:
        pass



