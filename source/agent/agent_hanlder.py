# agent/agent_handler.py

from rl.agents import DQNAgent
from rl.policy import Policy
from rl.memory import SequentialMemory
from tensorflow.keras import Optimizer, Model
from  tensorflow.keras.callbacks import Callback 
from typing import Optional
import random

from ..environment import TradingEnvironment

class AgentHandler():
    """
    """

    def __init__(self, model: Model, policy: Policy, nr_of_actions: int, optimizer: Optimizer) -> None:
        """
        """

        self.trained: bool = False
        self.agent: DQNAgent = DQNAgent(model, policy, memory = SequentialMemory(limit = 100000, window_length = 1),
                                        nb_actions = nr_of_actions, target_model_update = 1e-2)
        self.agent.compile(optimizer)

    def train_agent(self, environment: TradingEnvironment, nr_of_steps: int, steps_per_episode: int, 
                    callbacks: list[Callback] = [], weights_load_path: Optional[str] = None, 
                    weights_save_path: Optional[str] = None) -> dict:
        """
        """

        if weights_load_path is not None:
            self.agent.load_weights(weights_load_path)
        
        history = self.agent.fit(environment, nr_of_steps, callbacks = callbacks, 
                                 log_interval = steps_per_episode, nb_max_episode_steps = steps_per_episode)
        self.trained = True

        if weights_save_path is not None:
            self.agent.save_weights(weights_save_path)
        
        return history
    
    def test_agent(self, environment: TradingEnvironment, repeat: int = 1) -> dict:
        """
        """

        if not self.trained:
            return
        
        test_history = {}
        env_length = environment.get_environment_length()
        for i in range(repeat):
            test_history[i] = {}
            assets_values = []
            reward_values = []
            infos = []
            iterations = []
            done = False

            current_iteration = random.randint(0, int(env_length/2)) 
            environment.reset(current_iteration)
            state = environment.state
            trading_data = environment.get_trading_data()
            current_assets = trading_data.current_budget + trading_data.currently_invested
            iterations.append(current_iteration)
            assets_values.append(current_assets)
            reward_values.append(0)
            infos.append({})

            while(not done):
                next_action = self.agent.forward(state)
                state, reward, done, info = environment.step(next_action)

                if current_assets != info['current_budget'] + info['currently_invested']:
                    current_iteration = environment.current_iteration
                    current_assets = info['current_budget'] + info['currently_invested']
                    iterations.append(current_iteration)
                    assets_values.append(current_assets)
                    reward_values.append(reward)
                    infos.append(info)
            
            assets_values = assets_values / assets_values[0]
            solvency_coefficient = assets_values[-1] / (iterations[-1] - iterations[0])

            test_history[i]['assets_values'] = assets_values
            test_history[i]['reward_values'] = reward_values
            test_history[i]['infos'] = infos
            test_history[i]['iterations'] = iterations
            test_history[i]['solvency_coefficient'] = solvency_coefficient

            return test_history






    