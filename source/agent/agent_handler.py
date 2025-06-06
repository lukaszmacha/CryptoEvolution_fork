# agent/agent_handler_base.py

# global imports
import io
import logging
import random
from contextlib import redirect_stdout
from tensorflow.keras.callbacks import Callback
from typing import Any, Callable, Optional

# local imports
from source.agent import AgentBase, LearningStrategyHandlerBase, TestingStrategyHandlerBase
from source.environment import TradingEnvironment
from source.model import BluePrintBase

class AgentHandler():
    """"""

    def __init__(self, model_blue_print: BluePrintBase,
                 trading_environment: TradingEnvironment,
                 learning_strategy_handler: LearningStrategyHandlerBase,
                 testing_strategy_handler: TestingStrategyHandlerBase) -> None:
        """"""

        self.__trained: bool = False
        self.__learning_strategy_handler: LearningStrategyHandlerBase = learning_strategy_handler
        self.__testing_strategy_handler: TestingStrategyHandlerBase = testing_strategy_handler
        self.__trading_environment: TradingEnvironment = trading_environment
        self.__agent: AgentBase = learning_strategy_handler.create_agent(model_blue_print, trading_environment)

    def train_agent(self, nr_of_steps: int, nr_of_episodes: int, callbacks: Optional[list[Callback]] = None,
                    model_load_path: Optional[str] = None,
                    model_save_path: Optional[str] = None) -> tuple[list[str], list[dict]]:
        """"""

        if callbacks is None:
            callbacks = []

        self.__trading_environment.set_mode(TradingEnvironment.TRAIN_MODE)

        if model_load_path is not None:
            self.__agent.load_model(model_load_path)

        captured_output = io.StringIO()
        with redirect_stdout(captured_output): #TODO: Create an callback logger
            key, report_data = self.__learning_strategy_handler.fit(self.__agent, self.__trading_environment,
                                                                    nr_of_steps, nr_of_episodes, callbacks)

            for line in captured_output.getvalue().split('\n'):
                if line.strip():
                    logging.info(line)
        self.__trained = True

        if model_save_path is not None:
            self.__agent.save_model(model_save_path)

        return key, report_data

    def test_agent(self, repeat: int = 1) -> tuple[dict[int, list[str]], dict[int, list[dict[str, Any]]]]:
        """"""

        if not self.__trained:
            logging.error('Agent is not trained yet! Train the agent before testing.')
            return {}, {}

        self.__trading_environment.set_mode(TradingEnvironment.TEST_MODE)

        report_data = {}
        key = {}
        for i in range(repeat):
            env_length = self.__trading_environment.get_environment_length()
            window_size = self.__trading_environment.get_trading_consts().WINDOW_SIZE
            current_iteration = random.randint(window_size, int(env_length/2))
            self.__trading_environment.reset(current_iteration)
            key[i], report_data[i] = self.__testing_strategy_handler.evaluate(self.__agent,
                                                                              self.__trading_environment)

        return key, report_data

    def print_model_summary(self, print_function: Optional[Callable] = print) -> None:
        """"""

        self.__agent.print_summary(print_function = print_function)