# training/training_config.py

# global imports
from typing import Optional

# local imports
from source.agent import AgentHandler, LearningStrategyHandlerBase, TestingStrategyHandlerBase
from source.environment import LabelAnnotatorBase, RewardValidatorBase, SimpleLabelAnnotator, \
    PriceRewardValidator, TradingEnvironment
from source.model import BluePrintBase

class TrainingConfig():
    """"""

    def __init__(self, nr_of_steps: int, nr_of_episodes: int, model_blue_print: BluePrintBase,
                 data_path: str, initial_budget: float, max_amount_of_trades: int, window_size: int,
                 learning_strategy_handler: LearningStrategyHandlerBase,
                 testing_strategy_handler: TestingStrategyHandlerBase, sell_stop_loss: float = 0.8,
                 sell_take_profit: float = 1.2, buy_stop_loss: float = 0.8, buy_take_profit: float = 1.2,
                 penalty_starts: int = 0, penalty_stops: int = 10, static_reward_adjustment: float = 1,
                 repeat_test: int = 10, test_ratio: float = 0.2, validator: Optional[RewardValidatorBase] = None,
                 label_annotator: Optional[LabelAnnotatorBase] = None) -> None:
        """"""

        if validator is None:
            validator = PriceRewardValidator()

        if label_annotator is None:
            label_annotator = SimpleLabelAnnotator()

        # Training config
        self.nr_of_steps: int = nr_of_steps
        self.nr_of_episodes: int = nr_of_episodes
        self.repeat_test: int = repeat_test

        # Environment config
        self.__data_path: str = data_path
        self.__test_ratio = test_ratio
        self.__initial_budget: float = initial_budget
        self.__max_amount_of_trades: int = max_amount_of_trades
        self.__window_size: int = window_size
        self.__sell_stop_loss: float = sell_stop_loss
        self.__sell_take_profit: float = sell_take_profit
        self.__buy_stop_loss: float = buy_stop_loss
        self.__buy_take_profit: float = buy_take_profit
        self.__penalty_starts: int = penalty_starts
        self.__penalty_stops: int = penalty_stops
        self.__static_reward_adjustment: float = static_reward_adjustment
        self.__validator: RewardValidatorBase = validator
        self.__label_annotator: LabelAnnotatorBase = label_annotator

        # Agent config
        self.__model_blue_print: BluePrintBase = model_blue_print
        self.__learning_strategy_handler: LearningStrategyHandlerBase = learning_strategy_handler
        self.__testing_strategy_handler: TestingStrategyHandlerBase = testing_strategy_handler

    def __str__(self) -> str:
        """
        Returns a string representation of the configuration.

        Creates a formatted multi-line string containing all configuration
        parameters and their values for easy logging.

        Returns:
            str: Formatted string representation of the configuration.
        """

        return f"Training config:\n" \
                f"\tnr_of_steps: {self.nr_of_steps}\n" \
                f"\tnr_of_episodes: {self.nr_of_episodes}\n" \
                f"\trepeat_test: {self.repeat_test}\n" \
                f"\ttest_ratio: {self.__test_ratio}\n" \
                f"\tinitial_budget: {self.__initial_budget}\n" \
                f"\tmax_amount_of_trades: {self.__max_amount_of_trades}\n" \
                f"\twindow_size: {self.__window_size}\n" \
                f"\tsell_stop_loss: {self.__sell_stop_loss}\n" \
                f"\tsell_take_profit: {self.__sell_take_profit}\n" \
                f"\tbuy_stop_loss: {self.__buy_stop_loss}\n" \
                f"\tbuy_take_profit: {self.__buy_take_profit}\n" \
                f"\tpenalty_starts: {self.__penalty_starts}\n" \
                f"\tpenalty_stops: {self.__penalty_stops}\n" \
                f"\tstatic_reward_adjustment: {self.__static_reward_adjustment}\n" \
                f"\tvalidator: {self.__validator.__class__.__name__}\n" \
                f"\t\t{vars(self.__validator)}\n" \
                f"\tmodel_blue_print: {self.__model_blue_print.__class__.__name__}\n" \
                f"\t\t{vars(self.__model_blue_print)}\n" \
                f"\tlearning_strategy_handler: {self.__learning_strategy_handler.__class__.__name__}\n" \
                f"\t\t{vars(self.__learning_strategy_handler)}\n" \
                f"\ttesting_strategy_handler: {self.__testing_strategy_handler.__class__.__name__}\n" \
                f"\t\t{self.__testing_strategy_handler}\n"

    def instantiate_agent_handler(self) -> AgentHandler:
        """"""

        environment = TradingEnvironment(self.__data_path, self.__initial_budget, self.__max_amount_of_trades,
                                         self.__window_size, self.__validator, self.__label_annotator,
                                         self.__sell_stop_loss, self.__sell_take_profit, self.__buy_stop_loss,
                                         self.__buy_take_profit, self.__test_ratio, self.__penalty_starts,
                                         self.__penalty_stops, self.__static_reward_adjustment)

        return AgentHandler(self.__model_blue_print, environment, self.__learning_strategy_handler,
                            self.__testing_strategy_handler)