# agent/agent_handler.py

import random
import numpy as np
import logging
import io
import rl
from rl.agents import DQNAgent
import rl.agents
from rl.policy import Policy
from rl.memory import SequentialMemory
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import Callback
from typing import Optional
from contextlib import redirect_stdout

from source.environment import TradingEnvironment

class AgentHandler():
    """
    Responsible for encapsulating a DQNAgent along with its associated training and testing procedures.
    This class provides a simplified interface for managing deep reinforcement learning agent operations
    in the trading environment context.
    """

    def __init__(self, model: Model, policy: Policy, nr_of_actions: int, optimizer: Optimizer) -> None:
        """
        Initializes the AgentHandler with given model, policy and action space parameters.

        Parameters:
            model (Model): Keras model used by the agent to learn from environment.
            policy (Policy): Policy that determines action selection strategy.
            nr_of_actions (int): Number of possible actions agent can take.
            optimizer (Optimizer): Keras optimizer used for model training.
        """

        self.__trained: bool = False
        self.__agent: DQNAgent = rl.agents.DQNAgent(model, policy, memory = SequentialMemory(limit = 100000, window_length = 1),
                                                  nb_actions = nr_of_actions, target_model_update = 1e-2)
        self.__agent.compile(optimizer)

    def train_agent(self, environment: TradingEnvironment, nr_of_steps: int, steps_per_episode: int,
                    callbacks: list[Callback] = [], weights_load_path: Optional[str] = None,
                    weights_save_path: Optional[str] = None) -> dict:
        """
        Trains the agent on the provided environment.

        Parameters:
            environment (TradingEnvironment): Trading environment to train on.
            nr_of_steps (int): Total number of training steps.
            steps_per_episode (int): Maximum steps per episode.
            callbacks (list[Callback], optional): List of Keras callbacks for training.
            weights_load_path (str, optional): Path to load pre-trained weights.
            weights_save_path (str, optional): Path to save weights after training.

        Returns:
            dict: Dictionary containing training history metrics.
        """

        if weights_load_path is not None:
            self.__agent.load_weights(weights_load_path)

        captured_output = io.StringIO()
        with redirect_stdout(captured_output): #TODO: Create an callback logger
            history = self.__agent.fit(environment, nr_of_steps, callbacks = callbacks,
                                    log_interval = steps_per_episode, nb_max_episode_steps = steps_per_episode)

            for line in captured_output.getvalue().split('\n'):
                if line.strip():
                    logging.info(line)
        self.__trained = True

        if weights_save_path is not None:
            self.__agent.save_weights(weights_save_path)

        return history.history

    def test_agent(self, environment: TradingEnvironment, repeat: int = 1) -> dict:
        """
        Tests the trained agent on the provided environment.

        Testing involves running the agent on the environment from random starting points
        and recording the performance metrics like asset value changes and rewards.

        Parameters:
            environment (TradingEnvironment): Trading environment to test on.
            repeat (int, optional): Number of test episodes to run. Defaults to 1.

        Returns:
            dict: Dictionary containing test metrics including asset values, rewards,
                  and trading performance statistics for each test episode.
                  Returns empty dict if agent is not trained.
        """

        if not self.__trained:
            logging.error('Agent is not trained yet! Train the agent before testing.')
            return {}

        test_history = {}
        env_length = environment.get_environment_length()
        for i in range(repeat):
            test_history[i] = {}
            assets_values = []
            reward_values = []
            infos = []
            iterations = []
            done = False

            window_size = environment.get_trading_consts().WINDOW_SIZE
            current_iteration = random.randint(window_size, int(env_length/2))
            environment.reset(current_iteration)
            state = environment.state
            trading_data = environment.get_trading_data()
            current_assets = trading_data.current_budget + trading_data.currently_invested
            iterations.append(current_iteration)
            assets_values.append(current_assets)
            reward_values.append(0)
            infos.append({})

            while(not done):
                next_action = self.__agent.forward(state)
                state, reward, done, info = environment.step(next_action)

                if current_assets != info['current_budget'] + info['currently_invested'] or done:
                    current_iteration = environment.current_iteration
                    current_assets = info['current_budget'] + info['currently_invested']
                    iterations.append(current_iteration)
                    assets_values.append(current_assets)
                    reward_values.append(reward)
                    infos.append(info)

            solvency_coefficient = (assets_values[-1] - assets_values[0]) / (iterations[-1] - iterations[0])
            assets_values = (np.array(assets_values) / assets_values[0]).tolist()
            currency_prices = environment.get_data_for_iteration(['close'], iterations[0], iterations[-1])
            currency_prices = (np.array(currency_prices) / currency_prices[0]).tolist()

            test_history[i]['assets_values'] = assets_values
            test_history[i]['reward_values'] = reward_values
            test_history[i]['currency_prices'] = currency_prices
            test_history[i]['infos'] = infos
            test_history[i]['iterations'] = iterations
            test_history[i]['solvency_coefficient'] = solvency_coefficient

        return test_history

    def print_model_summary(self, print_function: Optional[callable] = print) -> None:
        """
        Prints the model summary using the provided print function.

        Parameters:
            print_function (callable, optional): Function to use for printing.
                                                Defaults to built-in print.
        """

        self.__agent.model.summary(print_fn = print_function)