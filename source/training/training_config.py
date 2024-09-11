# training/training_config.py

from rl.policy import Policy, BoltzmannQPolicy
from tensorflow.keras import Optimizer
from tensorflow.keras.optimizers.legacy import Adam

from ..environment.trading_environment import TradingEnvironment
from ..agent.agent_hanlder import AgentHandler
from ..model.model_blue_prints.base_blue_print import BaseBluePrint
from ..environment.reward_validator_base import RewardValidatorBase

class TrainingConfig():
    """
    """

    def __init__(self, nr_of_steps: int, nr_of_episodes: int, model_blue_print: BaseBluePrint, 
                 data_path: str, initial_budget: float, max_amount_of_trades: int, window_size: int,
                 validator: RewardValidatorBase, sell_stop_loss: float = 0.8, sell_take_profit: float = 1.2,
                 buy_stop_loss: float = 0.8, buy_take_profit: float = 1.2, penalty_starts: int = 0, 
                 penalty_stops: int = 10, static_reward_adjustment: float = 1, policy: Policy = BoltzmannQPolicy(),
                 optimizer: Optimizer = Adam(lr=1e-3), repeat_test: int = 10) -> None:
        """
        """

        # Training config
        self.nr_of_steps = nr_of_steps
        self.nr_of_episodes = nr_of_episodes
        self.repeat_test = repeat_test

        # Environment config
        self.__data_path: str = data_path
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
        
        # Agent config
        self.__model_blue_print: BaseBluePrint = model_blue_print
        self.__policy: Policy = policy
        self.__optimizer: Optimizer = optimizer

    def instantiate_environment(self) -> TradingEnvironment:
        """
        """

        self.__instantiated_environment = TradingEnvironment(self.__data_path,
                                                             self.__initial_budget,
                                                             self.__max_amount_of_trades, 
                                                             self.__window_size, 
                                                             self.__validator, 
                                                             self.__sell_stop_loss, 
                                                             self.__sell_take_profit, 
                                                             self.__buy_stop_loss, 
                                                             self.__buy_take_profit, 
                                                             self.__penalty_starts, 
                                                             self.__penalty_stops, 
                                                             self.__static_reward_adjustment)

        return self.__instantiated_environment

    def instantiate_agent(self) -> AgentHandler:
        """
        """

        if self.__instantiated_environment is None:
            raise RuntimeError("Environment not instantiated yet!")

        nr_of_actions = self.__instantiated_environment.action_space.n
        output_length = self.__instantiated_environment.observation_space.shape
        spatial_data_shape = self.__instantiated_environment.get_environment_spatial_data_dimension()
        model = self.__model_blue_print.instantiate_model(nr_of_actions, output_length, spatial_data_shape)
        self.__instantiated_agent = AgentHandler(model, self.__policy, nr_of_actions, self.__optimizer)

        return self.__instantiated_agent