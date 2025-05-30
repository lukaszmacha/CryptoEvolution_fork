# training/training_config.py

from rl.policy import Policy, BoltzmannQPolicy
from tensorflow.keras.optimizers import Optimizer, Adam

from ..environment.trading_environment import TradingEnvironment
from ..agent.agent_handler import AgentHandler
from ..model.model_blue_prints.base_blue_print import BaseBluePrint
from ..environment.reward_validator_base import RewardValidatorBase

class TrainingConfig():
    """
    Responsible for creating and configuring training environment and agent.

    This class encapsulates all configuration parameters needed for training
    and testing a trading agent, including environment setup, agent creation,
    and reward validation. It provides a centralized way to manage training
    parameters and instantiate required components.
    """

    def __init__(self, nr_of_steps: int, nr_of_episodes: int, model_blue_print: BaseBluePrint,
                 data_path: str, initial_budget: float, max_amount_of_trades: int, window_size: int,
                 validator: RewardValidatorBase, sell_stop_loss: float = 0.8, sell_take_profit: float = 1.2,
                 buy_stop_loss: float = 0.8, buy_take_profit: float = 1.2, penalty_starts: int = 0,
                 penalty_stops: int = 10, static_reward_adjustment: float = 1, policy: Policy = BoltzmannQPolicy(),
                 optimizer: Optimizer = Adam(learning_rate=1e-3), repeat_test: int = 10, test_ratio: float = 0.2) -> None:
        """
        Initializes the training configuration with provided parameters.

        Parameters:
            nr_of_steps (int): Total number of training steps.
            nr_of_episodes (int): Number of training episodes.
            model_blue_print (BaseBluePrint): Blueprint for creating the neural network model.
            data_path (str): Path to the training data file.
            initial_budget (float): Starting budget for the agent.
            max_amount_of_trades (int): Maximum number of trades allowed to be placed in the environment.
            window_size (int): Size of the observation window for market data.
            validator (RewardValidatorBase): Strategy for validating and calculating rewards.
            sell_stop_loss (float): Coefficient defining when to stop loss on sell positions.
            sell_take_profit (float): Coefficient defining when to take profit on sell positions.
            buy_stop_loss (float): Coefficient defining when to stop loss on buy positions.
            buy_take_profit (float): Coefficient defining when to take profit on buy positions.
            penalty_starts (int): Starting point (in trading periods without activity) that penalty should be applied from.
            penalty_stops (int): Ending point (in trading periods without activity) that penalty growth should be stopped at.
            static_reward_adjustment (float): Adjustment factor for rewards, used to penalize unwanted actions.
            policy (Policy): Policy for action selection during training.
            optimizer (Optimizer): Optimizer to be used for model compilation and training.
            repeat_test (int): Number of times to repeat testing for evaluation.
            test_ratio (float): Ratio of data to be used for testing vs training.
        """

        # Training config
        self.nr_of_steps = nr_of_steps
        self.nr_of_episodes = nr_of_episodes
        self.repeat_test = repeat_test

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
        self.__instantiated_environment: TradingEnvironment = None

        # Agent config
        self.__model_blue_print: BaseBluePrint = model_blue_print
        self.__policy: Policy = policy
        self.__optimizer: Optimizer = optimizer
        self.__instantiated_agent: AgentHandler = None

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
                f"\tpolicy: {self.__policy.__class__.__name__}\n" \
                f"\t\t{vars(self.__policy)}\n" \
                f"\toptimizer: {self.__optimizer.__class__.__name__}\n" \
                f"\t\t{self.__optimizer._hyper}\n"

    def instantiate_environment(self) -> TradingEnvironment:
        """
        Creates and returns a TradingEnvironment based on the configuration.

        Instantiates a new trading environment with the parameters specified
        in this config. Stores the created environment internally for later use
        when creating the agent.

        Returns:
            TradingEnvironment: Configured trading environment ready for training.
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
                                                             self.__test_ratio,
                                                             self.__penalty_starts,
                                                             self.__penalty_stops,
                                                             self.__static_reward_adjustment)

        return self.__instantiated_environment

    def instantiate_agent(self) -> AgentHandler:
        """
        Creates and returns an AgentHandler based on the configuration.

        Uses the model blueprint to create a neural network model with the correct
        input and output dimensions based on the environment's observation and action
        spaces. Then wraps this model in an AgentHandler with the specified policy
        and optimizer.

        Returns:
            AgentHandler: Configured agent handler ready for training.

        Raises:
            RuntimeError: If environment has not been instantiated first.
        """

        if self.__instantiated_environment is None:
            raise RuntimeError("Environment not instantiated yet!")

        observation_space_shape = self.__instantiated_environment.observation_space.shape
        nr_of_actions = self.__instantiated_environment.action_space.n
        spatial_data_shape = self.__instantiated_environment.get_environment_spatial_data_dimension()
        model = self.__model_blue_print.instantiate_model(observation_space_shape, nr_of_actions, spatial_data_shape)
        self.__instantiated_agent = AgentHandler(model, self.__policy, nr_of_actions, self.__optimizer)

        return self.__instantiated_agent