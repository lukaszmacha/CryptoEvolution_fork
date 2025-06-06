# agent/strategies/reinforcement_learning_agent.py

# global imports
import rl
from rl.memory import SequentialMemory
from typing import Optional, Any, Callable
from rl.agents import DQNAgent
from tensorflow.keras.optimizers import Optimizer
from rl.policy import Policy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model

# local imports
from source.agent import AgentBase
from source.agent import PerformanceTestable
from source.environment import TradingEnvironment

class ReinforcementLearningAgent(AgentBase, PerformanceTestable):
    """"""

    def __init__(self, model: Model, policy: Policy, optimizer: Optimizer) -> None:
        """"""

        memory = SequentialMemory(limit = 100000, window_length = 1)
        self.__DQNAgent: DQNAgent = rl.agents.DQNAgent(model, policy, memory = memory,
                                                       nb_actions = model.output_shape[-1],
                                                       target_model_update = 1e-2)
        self.__DQNAgent.compile(optimizer)

    def load_model(self, model_path: str) -> None:
        """"""

        self.__DQNAgent.load_weights(model_path)

    def save_model(self, model_path: str) -> None:
        """"""

        self.__DQNAgent.save_weights(model_path)

    def print_summary(self, print_function: Optional[Callable] = print) -> None:
        """"""

        self.__DQNAgent.model.summary(print_fn = print_function)

    def reinforcement_learning_fit(self, environment: TradingEnvironment, nr_of_steps: int,
                                   steps_per_episode: int, callbacks: list[Callback]) -> dict[str, Any]:
        """"""

        return self.__DQNAgent.fit(environment, nr_of_steps, callbacks = callbacks,
                                   log_interval = steps_per_episode,
                                   nb_max_episode_steps = steps_per_episode).history

    def perform(self, observation: list[float]) -> int:
        """"""

        return self.__DQNAgent.forward(observation)