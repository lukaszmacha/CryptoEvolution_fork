# agent/strategies/reinforcement_learning_strategy_handler.py

# global imports
from typing import Any
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.callbacks import Callback
from rl.policy import Policy, BoltzmannQPolicy

# local imports
from source.agent import LearningStrategyHandlerBase
from source.agent import AgentBase
from source.agent import ReinforcementLearningAgent
from source.environment import TradingEnvironment
from source.model import BluePrintBase

class ReinforcementLearningStrategyHandler(LearningStrategyHandlerBase):
    """"""

    PLOTTING_KEY: str = 'reinforcement_learning'

    def __init__(self, policy: Policy = BoltzmannQPolicy(),
                 optimizer: Optimizer = Adam(learning_rate = 0.001)) -> None:
        """"""

        self.__policy: Policy = policy
        self.__optimizer: Optimizer = optimizer

    def create_agent(self, model_blue_print: BluePrintBase,
                     trading_environment: TradingEnvironment) -> AgentBase:
        """"""

        observation_space_shape = trading_environment.observation_space.shape
        nr_of_actions = trading_environment.action_space.n
        spatial_data_shape = trading_environment.get_environment_spatial_data_dimension()
        model_adapter = model_blue_print.instantiate_model(observation_space_shape, nr_of_actions,
                                                           spatial_data_shape)
        return ReinforcementLearningAgent(model_adapter.get_model(), self.__policy, self.__optimizer)

    def fit(self, agent: ReinforcementLearningAgent, trading_environment: TradingEnvironment,
            nr_of_steps: int, nr_of_episodes: int, callbacks: list[Callback]) -> tuple[list[str], dict[str, Any]]:

        if not isinstance(agent, ReinforcementLearningAgent):
            raise TypeError("Agent must be an instance of ReinforcementLearningAgent.")

        steps_per_episode = nr_of_steps // nr_of_episodes
        return [ReinforcementLearningStrategyHandler.PLOTTING_KEY], \
            [agent.reinforcement_learning_fit(trading_environment,
                                             nr_of_steps, steps_per_episode, callbacks)]