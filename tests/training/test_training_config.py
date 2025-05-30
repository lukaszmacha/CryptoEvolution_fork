# tests/agent/test_training_config.py

import logging
import pandas as pd
import re
from unittest import TestCase
from unittest.mock import Mock, patch
from ddt import ddt, data, unpack
from typing import Any
from types import SimpleNamespace
from tensorflow.keras import Model

from source.training import TrainingConfig
from source.environment.mock_validator import MockRewardValidator
from source.model import MockBluePrint

INITIAL_BUDGET = 1000.0
MAX_AMOUNT_OF_TRADES = 5
WINDOW_SIZE = 48
MOCKED_CSV_DATA = pd.DataFrame(data={
    'low': [20000.0, 20500.0, 20100.0, 20100.0, 20000.0],
    'high': [20900.0, 20900.0, 21000.0, 20900.0, 21700.0],
    'open': [20050.0, 20600.0, 20400.0, 20800.0, 20200.0],
    'close': [20600.0, 20400.0, 20800.0, 20200.0, 20900.0],
    'volume': [1000.0, 1200.0, 1100.0, 1300.0, 900.0]
}, index = pd.DatetimeIndex(['2020-03-01', '2020-03-02', '2020-03-03',
                             '2020-03-04', '2020-03-05'], name='time'))
MEMORY_LOCATION_IDENTIFIER_REGEX = r'\'.*\': <.* at 0x[0-9a-fA-F]+>'

@ddt
class TrainingConfigTestCase(TestCase):
    """
    Test case for TrainingConfig class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        nr_of_steps = 2000
        nr_of_episodes = 100
        model_blue_print = MockBluePrint(Model())
        data_path = "mock/path/to/data/set"
        validator = MockRewardValidator(lambda: 0)

        self.__sut: TrainingConfig = TrainingConfig(nr_of_steps = nr_of_steps,
                                                    nr_of_episodes = nr_of_episodes,
                                                    model_blue_print = model_blue_print,
                                                    data_path = data_path,
                                                    initial_budget = INITIAL_BUDGET,
                                                    max_amount_of_trades = MAX_AMOUNT_OF_TRADES,
                                                    window_size = WINDOW_SIZE,
                                                    validator = validator)

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injections of certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.__sut.__dict__:
                if name in attribute_name:
                    setattr(self.__sut, attribute_name, value)

    @data(
        ({
            'nr_of_steps': 3000,
            'nr_of_episodes': 200,
            'initial_budget': 1500.0,
            'max_amount_of_trades': 10,
            'window_size': 72
        },
            "Training config:\n"
            "\tnr_of_steps: 3000\n"
            "\tnr_of_episodes: 200\n"
            "\trepeat_test: 10\n"
            "\ttest_ratio: 0.2\n"
            "\tinitial_budget: 1500.0\n"
            "\tmax_amount_of_trades: 10\n"
            "\twindow_size: 72\n"
            "\tsell_stop_loss: 0.8\n"
            "\tsell_take_profit: 1.2\n"
            "\tbuy_stop_loss: 0.8\n"
            "\tbuy_take_profit: 1.2\n"
            "\tpenalty_starts: 0\n"
            "\tpenalty_stops: 10\n"
            "\tstatic_reward_adjustment: 1\n"
            "\tvalidator: MockRewardValidator\n"
            "\t\t{}\n"
            "\tmodel_blue_print: MockBluePrint\n"
            "\t\t{}\n"
            "\tpolicy: BoltzmannQPolicy\n"
            "\t\t{'tau': 1.0, 'clip': (-500.0, 500.0)}\n"
            "\toptimizer: Adam\n"
            "\t\t{'learning_rate': 0.001, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999}\n")
    )
    @unpack
    def test_training_config___str__(self, params_to_update: dict[str, Any], expected_serialization_str: str) -> None:
        """
        Tests TrainingConfig's __str__ functionality.

        Verifies that the string representation of the TrainingConfig object is correct
        after updating various parameters. The test sanitizes memory location identifiers
        to ensure consistent comparison.

        Parameters:
            params_to_update: Dictionary with parameters to update in the training config.
            expected_serialization_str: Expected string representation after updates.

        Asserts:
            The sanitized string representation matches the expected string.
        """

        logging.info("Attempting to serialize TrainingConfig.")
        self.__update_sut(**params_to_update)
        result = str(self.__sut)
        sanitized_result = re.sub(MEMORY_LOCATION_IDENTIFIER_REGEX, '', result)

        logging.info("Validating expected result.")
        self.assertEqual(sanitized_result, expected_serialization_str)

    @patch('pandas.read_csv', new_callable = Mock)
    @data(
        ({}, SimpleNamespace(
            INITIAL_BUDGET = INITIAL_BUDGET,
            MAX_AMOUNT_OF_TRADES = MAX_AMOUNT_OF_TRADES,
            WINDOW_SIZE = WINDOW_SIZE,
            SELL_STOP_LOSS = 0.8,
            SELL_TAKE_PROFIT = 1.2,
            BUY_STOP_LOSS = 0.8,
            BUY_TAKE_PROFIT = 1.2,
            STATIC_REWARD_ADJUSTMENT = 1,
            PENALTY_STARTS = 0,
            PENALTY_STOPS = 10)
        )
    )
    @unpack
    def test_training_config_instantiate_environment(self, params_to_update: dict[str, Any],
                                                     expected_training_consts: SimpleNamespace,
                                                     mock_pd_read_csv: Mock) -> None:
        """
        Tests TrainingConfig's instantiate_environment functionality.

        Verifies that instantiate_environment correctly creates a TradingEnvironment
        with the expected configuration constants. Uses a mock for pandas.read_csv
        to provide test data.

        Parameters:
            params_to_update: Dictionary with parameters to update in the training config.
            expected_trading_consts: SimpleNamespace containing expected trading constants.
            mock_pd_read_csv: Mock for pandas.read_csv function.

        Asserts:
            All trading constants in the instantiated environment match the expected values.
        """

        logging.info("Attempting to instantiate environment from TrainingConfig.")
        self.__update_sut(**params_to_update)

        mock_pd_read_csv.return_value = MOCKED_CSV_DATA
        result = self.__sut.instantiate_environment()

        logging.info("Validating returned environment.")
        for const_name, expected_value in expected_training_consts.__dict__.items():
            self.assertEqual(getattr(result.get_trading_consts(), const_name), expected_value)

    def test_training_config_instantiate_agent__environment_not_instantiated(self) -> None:
        """
        Tests TrainingConfig's instantiate_agent error handling when environment is not instantiated.

        Verifies that attempting to instantiate an agent without first instantiating
        the environment raises a RuntimeError. This enforces the required sequence
        of environment creation before agent creation.

        Asserts:
            A RuntimeError is raised when instantiate_agent is called before instantiate_environment.
        """

        logging.info("Attempting to instantiate agent from TrainingConfig without environment.")

        logging.info("Validating expected errors to be raised.")
        with self.assertRaises(RuntimeError) as context:
            self.__sut.instantiate_agent()