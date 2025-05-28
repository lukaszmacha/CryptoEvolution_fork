# plotting/plot_testing_history_responsibility_chain.py

import matplotlib.pyplot as plt
import numpy as np

from .plot_responsibility_chain_base import PlotResponsibilityChainBase

class PlotTestingHistoryResponsibilityChain(PlotResponsibilityChainBase):
    """
    Responsibility chain handler for plotting testing history data.

    This class implements a specific handler in the plotting chain that processes
    and visualizes testing results from the trading agent. It plots the growth
    of assets value compared to the underlying currency price, showing how the
    agent's performance compares to simply holding the currency.
    """

    def __init__(self, window_size: int = 5) -> None:
        """
        Initializes the testing history plot handler.

        Parameters:
            window_size (int): Size of the moving average window used for smoothing
                              the assets value curve. Defaults to 5.
        """

        self.__window_size = window_size
        super().__init__()

    def _can_plot(self, key: str) -> bool:
        """
        Determines if this handler can process the plot request.

        Checks if the key identifies a testing history plot type that this
        handler can process.

        Parameters:
            key (str): String identifier of the plot type.

        Returns:
            bool: True if this handler can process the plot request, False otherwise.
        """

        return key == 'testing_history'

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Generates a plot comparing asset performance to currency price.

        Creates a visualization that shows how the agent's portfolio value changed
        over time compared to the underlying currency price. Also shows a moving
        average of the assets value for trend clarity.

        Parameters:
            plot_data (dict): Dictionary containing testing history data with keys:
                             'assets_values', 'currency_prices', 'iterations',
                             and 'solvency_coefficient'.

        Returns:
            plt.Axes: Matplotlib Axes object containing the generated plot.
        """

        assets_values = plot_data['assets_values']
        currency_prices = plot_data['currency_prices']
        steps = plot_data['iterations']
        solvency_coefficient = plot_data['solvency_coefficient']
        adjusted_window_size = min(self.__window_size, len(assets_values))
        filter = np.ones(adjusted_window_size) / adjusted_window_size
        avg_assets_values = np.convolve(assets_values, filter, mode = 'same')

        plt.figure(figsize = (8, 6))
        plt.plot(range(steps[0], steps[0] + len(currency_prices)), currency_prices, label = 'Currency price growth', color = 'green')
        plt.plot(steps, assets_values, label = 'Assets value growth', color = 'blue')
        plt.plot(steps, avg_assets_values,
                 label = f'{adjusted_window_size}-step moving average of assets value growth', color = 'red')
        plt.title(f'Testing history with solvency {solvency_coefficient}')
        plt.xlabel('Number of steps')
        plt.ylabel('Currency price and assets value growth')
        plt.legend()

        return plt.gca()
