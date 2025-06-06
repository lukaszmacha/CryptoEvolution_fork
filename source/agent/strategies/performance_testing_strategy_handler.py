# agent/strategies/performance_testing_strategy_handler.py

# global imports
from typing import Any
import random
import numpy as np

# local imports
from source.environment import TradingEnvironment
from source.agent import TestingStrategyHandlerBase
from source.agent import PerformanceTestable

class PerformanceTestingStrategyHandler(TestingStrategyHandlerBase):
    """"""

    PLOTTING_KEY: str = 'performance_testing'

    def evaluate(self, testable_agent: PerformanceTestable, environment: TradingEnvironment) -> \
        tuple[list[str], list[dict[str, Any]]]:
        """"""

        history = {}
        assets_values = []
        reward_values = []
        infos = []
        iterations = []
        done = False

        state = environment.state
        trading_data = environment.get_trading_data()
        current_assets = trading_data.current_budget + trading_data.currently_invested
        iterations.append(current_iteration)
        assets_values.append(current_assets)
        reward_values.append(0)
        infos.append({})

        while(not done):
            next_action = testable_agent.perform(state)
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

        history['assets_values'] = assets_values
        history['reward_values'] = reward_values
        history['currency_prices'] = currency_prices
        history['infos'] = infos
        history['iterations'] = iterations
        history['solvency_coefficient'] = solvency_coefficient

        return PerformanceTestingStrategyHandler.PLOTTING_KEY, history