# tests/agent/test_agent_handler.py

import logging
from unittest import TestCase
from unittest.mock import Mock, patch
from ddt import ddt, data, unpack
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer
from rl.policy import Policy
from rl.agents import DQNAgent
from types import SimpleNamespace

from source.agent import AgentHandler
from source.environment import TradingEnvironment

@ddt
class AgentHandlerTestCase(TestCase):
    """
    Test case for AgentHandler class. Stores all the test cases
    and allows for convenient test case execution.
    """

    @patch('rl.agents.DQNAgent', new_callable = Mock)
    def setUp(self, mock_dqn_agent) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        mock_dqn_agent.return_value = Mock(spec = DQNAgent)
        model = Model()
        policy = Policy()
        nr_of_actions = 5
        optimizer = Optimizer(name = 'adam')

        self.__sut: AgentHandler = AgentHandler(model, policy, nr_of_actions, optimizer)
        self.__mocked_dqn_agent = mock_dqn_agent.return_value
        self.__mocked_dqn_agent.compile.assert_called_once_with(optimizer)

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


    def test_agent_handler_train_agent(self) -> None:
        """
        Tests AgentHandler's train_agent functionality.

        Verifies that the train_agent method correctly loads weights, fits
        the model to the environment, and saves the model weights after training.
        Mocks are used to isolate the test from actual model training.

        Asserts:
            The appropriate methods on the DQNAgent are called with correct parameters.
        """

        logging.info("Attempt to train agent with loading weights.")
        mock_environment = Mock(spec = TradingEnvironment)
        nr_of_steps = 1000
        steps_per_episode = 50
        weights_load_path = "mock/path/to/load/weights.h5"
        weights_save_path = "mock/path/to/save/weights.h5"
        callbacks = []

        self.__sut.train_agent(mock_environment, nr_of_steps, steps_per_episode,
                                callbacks = callbacks,
                                weights_load_path = weights_load_path,
                                weights_save_path = weights_save_path)

        logging.info("Validating expected calls.")
        self.__mocked_dqn_agent.load_weights.assert_called_once_with(weights_load_path)
        self.__mocked_dqn_agent.fit.assert_called_once_with(mock_environment, nr_of_steps,
                                                            callbacks = callbacks,
                                                            log_interval = steps_per_episode,
                                                            nb_max_episode_steps = steps_per_episode)
        self.__mocked_dqn_agent.save_weights.assert_called_once_with(weights_save_path)

    def test_agent_handler_test_agent__agent_not_fitted(self) -> None:
        """
        Tests AgentHandler's test_agent functionality when agent is not trained.

        Verifies that the test_agent method correctly handles the case where the
        agent has not been trained yet. In this case, no actions should be taken
        and the forward method should not be called.

        Asserts:
            The DQNAgent's forward method is not called when the agent is untrained.
        """

        logging.info("Attempt to test agent without training.")
        self.__update_sut(_AgentHandler__trained = False)
        mock_environment = Mock(spec = TradingEnvironment)
        repeat = 1

        self.__sut.test_agent(mock_environment, repeat)

        logging.info("Validating expected calls.")
        self.__mocked_dqn_agent.forward.assert_not_called()

    def test_agent_handler_test_agent__agent_fitted_properly(self) -> None:
        """
        Tests AgentHandler's test_agent functionality when agent is properly trained.

        Verifies that the test_agent method correctly interacts with the environment
        when the agent has been trained. It should set up the environment, call the
        agent's forward method to get actions, and process state transitions until
        the episode is complete.

        Asserts:
            The DQNAgent's forward method is called with the correct state.
        """

        logging.info("Attempt to test agent without training.")
        self.__update_sut(_AgentHandler__trained = True)
        current_state = [0] * 48
        current_budget = 1000
        currently_invested = 0

        mock_environment = Mock(spec = TradingEnvironment)
        mock_environment.state = current_state
        mock_environment.current_iteration = 1
        mock_environment.get_environment_length.return_value = 100
        mock_environment.get_trading_consts.return_value = SimpleNamespace(WINDOW_SIZE = 48)
        mock_environment.get_trading_data.return_value = SimpleNamespace(current_budget = current_budget,
                                                                         currently_invested = currently_invested)
        mock_environment.step.return_value = (current_state, 0, True, {'current_budget': current_budget,
                                                                       'currently_invested': currently_invested})
        mock_environment.get_data_for_iteration.return_value = [80000, 80000]
        repeat = 1

        self.__mocked_dqn_agent.forward.return_value = 1
        self.__sut.test_agent(mock_environment, repeat)

        logging.info("Validating expected calls.")
        self.__mocked_dqn_agent.forward.assert_called_with(current_state)