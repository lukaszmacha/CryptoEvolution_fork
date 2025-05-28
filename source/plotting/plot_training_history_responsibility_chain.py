# plotting/plot_training_history_responsibility_chain.py

import matplotlib.pyplot as plt
import numpy as np

from .plot_responsibility_chain_base import PlotResponsibilityChainBase

class PlotTrainingHistoryResponsibilityChain(PlotResponsibilityChainBase):
    """
    Responsibility chain handler for plotting training history data.

    This class implements a specific handler in the plotting chain that processes
    and visualizes training results from the trading agent. It plots the reward
    history over training steps, showing how the agent's performance evolved
    during the training process.
    """

    def __init__(self, window_size: int = 5) -> None:
        """
        Initializes the training history plot handler.

        Parameters:
            window_size (int): Size of the moving average window used for smoothing
                              the reward curve. Defaults to 5.
        """

        self.__window_size = window_size
        super().__init__()

    def _can_plot(self, key: str) -> bool:
        """
        Determines if this handler can process the plot request.

        Checks if the key identifies a training history plot type that this
        handler can process.

        Parameters:
            key (str): String identifier of the plot type.

        Returns:
            bool: True if this handler can process the plot request, False otherwise.
        """

        return key == 'training_history'

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates a plot showing reward progression during training.

        Creates a visualization that shows how the episode rewards changed
        over training steps. Also shows a moving average of the rewards
        for better trend visibility.

        Parameters:
            plot_data (dict): Dictionary containing training history data with keys:
                             'nb_steps' for step counts and 'episode_reward' for
                             corresponding rewards.

        Returns:
            plt.Axes: Matplotlib Axes object containing the generated plot.
        """

        steps = [0] + plot_data['nb_steps']
        reward = [0] + plot_data['episode_reward']
        adjusted_window_size = min(self.__window_size, len(reward))
        filter = np.ones(adjusted_window_size) / adjusted_window_size
        avg_reward = np.convolve(reward, filter, mode = 'same')

        plt.figure(figsize=(8, 6))
        plt.plot(steps, reward, label = 'Episode reward', color = 'blue')
        plt.plot(steps, avg_reward,
                 label = f'{adjusted_window_size}-step moving average of episode reward', color = 'red')
        plt.title('Training history')
        plt.xlabel('Number of steps')
        plt.ylabel('Reward')
        plt.legend()

        return plt.gca()
