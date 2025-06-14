# environment/trading environment.py

# global imports
from gym import Env
from gym.spaces import Discrete, Box
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import math
import random
from types import SimpleNamespace
from typing import Optional
import copy
from tensorflow.keras.utils import to_categorical
import logging

# local imports
from source.environment import Broker
from source.environment import RewardValidatorBase
from source.environment import LabelAnnotatorBase

class TradingEnvironment(Env):
    """
    Implements stock market environment that actor can perform actions (place orders) in.
    It is used to train Neural Network models with reinforcement learning approach. Can be
    configure to award points and impose a penalty in a several way.
    """

    TRAIN_MODE = 'train'
    TEST_MODE = 'test'

    def __init__(self, data_path: str, initial_budget: float, max_amount_of_trades: int, window_size: int,
                 validator: RewardValidatorBase, label_annotator: LabelAnnotatorBase, sell_stop_loss: float,
                 sell_take_profit: float, buy_stop_loss: float, buy_take_profit: float, test_ratio: float = 0.2,
                 penalty_starts: int = 0, penalty_stops: int = 10, static_reward_adjustment: float = 1) -> None:
        """
        Class constructor. Allows to define all crucial constans, reward validation methods,
        environmental penalty policies, etc.

        Parameters:
            data_path (str): Path to CSV data that should be used as enivronmental stock market.
            initial_budget (float): Initial budget constant for trader to start from.
            max_amount_of_trades (int): Max amount of trades that can be ongoing at the same time.
                Seting this constant prevents traders from placing orders randomly and defines
                amount of money that can be assigned to a single order at certain iteration.
            window_size (int): Constant defining how far in the past trader will be able to look
                into at certain iteration.
            validator (RewardValidatorBase): Validator implementing policy used to award points
                for closed trades.
            label_annotator (LabelAnnotatorBase): Annotator implementing policy used to label
                data with target values. It is used to provide supervised agents with information
                about what is the target class value for certain iteration.
            sell_stop_loss (float): Constant used to define losing boundary at which sell order
                (short) is closed.
            sell_take_profit (float): Constant used to define winning boundary at which sell order
                (short) is closed.
            buy_stop_loss (float): Constant used to define losing boundary at which buy order
                (long) is closed.
            buy_take_profit (float): Constant used to define winning boundary at which buy order
                (long) is closed.
            test_ratio (float): Ratio of data that should be used for testing purposes.
            penalty_starts (int): Constant defining how many trading periods can trader go without placing
                an order until penalty is imposed. Penalty at range between start and stop constant
                is calculated as percentile of positive reward, and subtracted from the actual reward.
            penalty_stops (int): Constant defining at which trading period penalty will no longer be increased.
                Reward for trading periods exceeding penalty stop constant will equal minus static reward adjustment.
            static_reward_adjustment (float): Constant use to penalize trader for bad choices or
                reward it for good one.
        """

        if test_ratio < 0.0 or test_ratio >= 1.0:
            raise ValueError(f"Invalid test_ratio: {test_ratio}. It should be in range [0, 1).")

        self.__data: dict[pd.DataFrame, pd.DataFrame] = self.__load_data(data_path, test_ratio)
        self.__mode = TradingEnvironment.TRAIN_MODE
        self.__broker: Broker = Broker()
        self.__validator: RewardValidatorBase = validator
        self.__label_annotator: LabelAnnotatorBase = label_annotator

        self.__trading_data: SimpleNamespace = SimpleNamespace()
        self.__trading_data.current_budget: float = initial_budget
        self.__trading_data.currently_invested: float = 0
        self.__trading_data.no_trades_placed_for: int = 0
        self.__trading_data.currently_placed_trades: int = 0

        self.__trading_consts = SimpleNamespace()
        self.__trading_consts.INITIAL_BUDGET: float = initial_budget
        self.__trading_consts.MAX_AMOUNT_OF_TRADES: int = max_amount_of_trades
        self.__trading_consts.WINDOW_SIZE: int = window_size
        self.__trading_consts.SELL_STOP_LOSS: float = sell_stop_loss
        self.__trading_consts.SELL_TAKE_PROFIT: float = sell_take_profit
        self.__trading_consts.BUY_STOP_LOSS: float = buy_stop_loss
        self.__trading_consts.BUY_TAKE_PROFIT: float = buy_take_profit
        self.__trading_consts.STATIC_REWARD_ADJUSTMENT: float = static_reward_adjustment
        self.__trading_consts.PENALTY_STARTS: int = penalty_starts
        self.__trading_consts.PENALTY_STOPS: int = penalty_stops
        self.__trading_consts.PROFITABILITY_FUNCTION = lambda x: -1.0 * math.exp(-x + 1) + 1
        self.__trading_consts.PENALTY_FUNCTION = lambda x: \
            min(1, 1 - math.tanh(-3.0 * (x - penalty_stops) / (penalty_stops - penalty_starts)))
        self.__trading_consts.OUTPUT_CLASSES: int = vars(self.__label_annotator.get_output_classes())

        self.current_iteration: int = self.__trading_consts.WINDOW_SIZE
        self.state: list[float] = self.__prepare_state_data()
        self.action_space: Discrete = Discrete(3)
        self.observation_space: Box = Box(low = np.ones(len(self.state)) * -3,
                                          high = np.ones(len(self.state)) * 3,
                                          dtype=np.float64)

    def __load_data(self, data_path: str, test_size: float) -> dict[pd.DataFrame, pd.DataFrame]:
        """
        Loads data from CSV file and splits it into training and testing sets based on the
        specified test size ratio.

        Parameters:
            data_path (str): Path to the CSV file containing the stock market data.
            test_size (float): Ratio of the data to be used for testing.

        Returns:
            (dict[pd.DataFrame, pd.DataFrame]): Dictionary containing training and testing data frames.
        """

        data_frame = pd.read_csv(data_path)
        dividing_index = int(len(data_frame) * (1 - test_size))

        return {
            TradingEnvironment.TRAIN_MODE: data_frame.iloc[:dividing_index].reset_index(drop=True),
            TradingEnvironment.TEST_MODE: data_frame.iloc[dividing_index:].reset_index(drop=True)
        }

    def __prepare_labeled_data(self) -> pd.DataFrame:
        """"""

        new_rows = []
        for i in range(self.current_iteration, self.get_environment_length() - 1):
            data_row = self.__prepare_state_data(slice(i - self.__trading_consts.WINDOW_SIZE, i), include_trading_data = False)
            new_rows.append(data_row)

        logging.info(f"New Rows Count: {len(self.__data[self.__mode])}")
        new_data = pd.DataFrame(new_rows, columns=[f"feature_{i}" for i in range(len(new_rows[0]))])
        logging.info(f"New Data Shape: {new_data.shape}")
        labels = self.__label_annotator.annotate(self.__data[self.__mode]).shift(-self.current_iteration)
        logging.info(f"Labels NaN Count: {labels.shape}")

        return new_data, labels.dropna()

    def __prepare_state_data(self, index: Optional[slice] = None, include_trading_data: bool = True) -> list[float]:
        """
        Calculates state data as a list of floats representing current iteration's observation.
        Observations contains all input data refined to window size and couple of coefficients
        giving an insight into current budget and orders situation.

        Returns:
           (list[float]): List with current observations for environment.
        """

        if index is None:
            index = slice(self.current_iteration - self.__trading_consts.WINDOW_SIZE, self.current_iteration)

        current_market_data = self.__data[self.__mode].iloc[index]
        current_market_data_no_index = current_market_data.select_dtypes(include = [np.number])
        normalized_current_market_data_values = pd.DataFrame(MinMaxScaler().fit_transform(current_market_data_no_index),
                                                             columns = current_market_data_no_index.columns).values
        current_marked_data_list = normalized_current_market_data_values.ravel().tolist()

        if include_trading_data:
            current_normalized_budget = 1.0 * self.__trading_data.current_budget / self.__trading_consts.INITIAL_BUDGET
            current_profitability_coeff = self.__trading_consts.PROFITABILITY_FUNCTION(current_normalized_budget)
            current_trades_occupancy_coeff = 1.0 * self.__trading_data.currently_placed_trades  / self.__trading_consts.MAX_AMOUNT_OF_TRADES
            current_no_trades_penalty_coeff = self.__trading_consts.PENALTY_FUNCTION(self.__trading_data.no_trades_placed_for)
            current_inner_state_list = [current_profitability_coeff, current_trades_occupancy_coeff, current_no_trades_penalty_coeff]
            current_marked_data_list += current_inner_state_list

        return current_marked_data_list

    def set_mode(self, mode: str) -> None:
        """
        Sets the mode of the environment to either TRAIN_MODE or TEST_MODE.

        Parameters:
            mode (str): Mode to set for the environment.

        Raises:
            ValueError: If the provided mode is not valid.
        """

        if mode not in [TradingEnvironment.TRAIN_MODE, TradingEnvironment.TEST_MODE]:
            raise ValueError(f"Invalid mode: {mode}. Use TradingEnvironment.TRAIN_MODE or TradingEnvironment.TEST_MODE.")
        self.__mode = mode

    def get_mode(self) -> str:
        """
        Mode getter.

        Returns:
            (str): Current mode of the environment.
        """

        return copy.copy(self.__mode)

    def get_trading_data(self) -> SimpleNamespace:
        """
        Trading data getter.

        Returns:
            (SimpleNamespace): Copy of the namespace with all trading data.
        """

        return copy.copy(self.__trading_data)

    def get_trading_consts(self) -> SimpleNamespace:
        """
        Trading constants getter.

        Returns:
            (SimpleNamespace): Copy of the namespace with all trading constants.
        """

        return copy.copy(self.__trading_consts)

    def get_broker(self) -> Broker:
        """
        Broker getter.

        Returns:
            (Broker): Copy of the broker used by environment.
        """

        return copy.copy(self.__broker)

    def get_environment_length(self) -> int:
        """
        Environment length getter.

        Returns:
            (Int): Length of environment.
        """

        return len(self.__data[self.__mode])

    def get_environment_spatial_data_dimension(self) -> tuple[int, int]:
        """
        Environment spatial data dimensionality getter.

        Returns:
            (Int): Dimension of spatial data in environment.
        """

        return (self.__trading_consts.WINDOW_SIZE, self.__data[self.__mode].shape[1] - 1)

    def get_labeled_data(self) -> tuple[np.ndarray, np.ndarray]:
        """"""

        input_data, output_data = self.__prepare_labeled_data()
        logging.info(f"Here Input data shape: {input_data.shape}, Output data shape: {output_data.shape}")
        input_data = np.expand_dims(np.array(input_data), axis = 1)
        output_data = to_categorical(np.array(output_data),
                                     num_classes = len(self.__trading_consts.OUTPUT_CLASSES))
        return copy.copy((input_data, output_data))

    def get_data_for_iteration(self, columns: list[str], start: int, stop: int, step: int = 1) -> list[float]:
        """
        Data for certain iterations getter.

        Parameters:
            columns (list[str]): List of column names to extract from data.
            start (int): Start iteration index.
            stop (int): Stop iteration index.
            step (int): Step between iterations. Default is 1.

        Returns:
            (list[float]): Copy of part of data with specified columns
                over specified iterations.
        """

        return copy.copy(self.__data[self.__mode].loc[start:stop:step, columns].values.ravel().tolist())

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        """
        Performs specified action on environment. It results in generation of the new
        observations. This function causes trades to be handled, reward to be calculated and
        environment to be updated.

        Parameters:
            action (int): Number specifing action. Possible values are 0 for buy action,
                1 for wait action and 2 for sell action.

        Returns:
            (tuple[list[float], float, bool, dict]): Tuple containing next observation
                state, reward, finish indication and additional info dictionary.
        """

        self.current_iteration += 1
        self.state = self.__prepare_state_data()

        close_changes = self.__data[self.__mode].iloc[self.current_iteration - 2 : self.current_iteration]['close'].values
        stock_change_coeff = 1 + (close_changes[1] - close_changes[0]) / close_changes[0]
        closed_orders= self.__broker.update_orders(stock_change_coeff)

        reward = self.__validator.validate_orders(closed_orders)
        self.__trading_data.currently_placed_trades -= len(closed_orders)
        self.__trading_data.current_budget += np.sum([trade.current_value for trade in closed_orders])
        self.__trading_data.currently_invested -= np.sum([trade.initial_value for trade in closed_orders])

        number_of_possible_trades = self.__trading_consts.MAX_AMOUNT_OF_TRADES - self.__trading_data.currently_placed_trades
        money_to_trade = 0
        if number_of_possible_trades > 0:
            money_to_trade = 1.0 / number_of_possible_trades * self.__trading_data.current_budget

        if action == 0:
            is_buy_order = True
            stop_loss = self.__trading_consts.SELL_STOP_LOSS
            take_profit = self.__trading_consts.SELL_TAKE_PROFIT
        elif action == 2:
            is_buy_order = False
            stop_loss = self.__trading_consts.BUY_STOP_LOSS
            take_profit = self.__trading_consts.BUY_TAKE_PROFIT

        if action != 1:
            if number_of_possible_trades > 0:
                self.__trading_data.current_budget -= money_to_trade
                self.__trading_data.currently_invested += money_to_trade
                self.__broker.place_order(money_to_trade, is_buy_order, stop_loss, take_profit)
                self.__trading_data.currently_placed_trades += 1
                self.__trading_data.no_trades_placed_for = 0
                reward += self.__trading_consts.STATIC_REWARD_ADJUSTMENT
            else:
                self.__trading_data.no_trades_placed_for += 1
                reward -= self.__trading_consts.STATIC_REWARD_ADJUSTMENT
        else:
            self.__trading_data.no_trades_placed_for += 1
            if number_of_possible_trades == 0:
                reward += self.__trading_consts.STATIC_REWARD_ADJUSTMENT

        if number_of_possible_trades > 0:
            reward *= (1 - self.__trading_consts.PENALTY_FUNCTION(self.__trading_data.no_trades_placed_for)) \
                      if reward > 0 else 1
            if self.__trading_consts.PENALTY_STOPS < self.__trading_data.no_trades_placed_for:
                reward -= self.__trading_consts.STATIC_REWARD_ADJUSTMENT

        if (self.current_iteration >= self.get_environment_length() - 1 or
            self.__trading_data.current_budget  > 10 * self.__trading_consts.INITIAL_BUDGET or
            (self.__trading_data.current_budget + self.__trading_data.currently_invested) / self.__trading_consts.INITIAL_BUDGET < 0.8):
            done = True
        else:
            done = False

        info = {'coeff': stock_change_coeff,
                'iteration': self.current_iteration,
                'number_of_closed_orders': len(closed_orders),
                'money_to_trade': money_to_trade,
                'action': action,
                'current_budget': self.__trading_data.current_budget,
                'currently_invested': self.__trading_data.currently_invested,
                'no_trades_placed_for': self.__trading_data.no_trades_placed_for,
                'currently_placed_trades': self.__trading_data.currently_placed_trades}

        return self.state, reward, done, info

    def render(self) -> None:
        """
        Renders environment visualization. Will be implemented later.
        """

        #TODO: Visualization to be implemented
        pass

    def reset(self, randkey: Optional[int] = None) -> list[float]:
        """
        Resets environment. Used typically if environemnt is finished,
        i.e. when ther is no more steps to be taken within environemnt
        or finish conditions are fulfilled.

        Parameters:
            randkey (Optional[int]): Value indicating what iteration
                should be trated as starting point after reset.

        Returns:
            (list[float]): Current iteration observation state.
        """

        if randkey is None:
            randkey = random.randint(self.__trading_consts.WINDOW_SIZE, self.get_environment_length() - 1)
        self.__trading_data.current_budget = self.__trading_consts.INITIAL_BUDGET
        self.__trading_data.currently_invested = 0
        self.__trading_data.no_trades_placed_for = 0
        self.__trading_data.currently_placed_trades = 0
        self.__broker.reset()
        self.current_iteration = randkey
        self.state = self.__prepare_state_data()

        return self.state