# tests/plotting/mock_plot_responsibility_chain.py

import matplotlib.pyplot as plt

from source.plotting import PlotResponsibilityChainBase

class MockPlotResponsibilityChain(PlotResponsibilityChainBase):
    """
    Mock implementation of the PlotResponsibilityChainBase for testing purposes.

    This class provides a simple implementation that returns a pre-configured
    matplotlib Axes object rather than generating a plot. It's primarily used
    in tests to isolate the plotting chain logic from the actual plotting
    implementation.
    """

    def __init__(self, axes: plt.Axes, key: str) -> None:
        """
        Initializes the mock plot responsibility chain with a pre-configured axes.

        Parameters:
            axes (plt.Axes): The matplotlib Axes object to be returned when _plot is called.
            key (str): The key string that this handler will respond to.
        """

        self.__axes = axes
        self.__key = key
        super().__init__()

    def _can_plot(self, key: str) -> bool:
        """
        Determines if this handler can process the plot request.

        Checks if the input key matches the key specified during initialization.

        Parameters:
            key (str): String identifier of the plot type.

        Returns:
            bool: True if the key matches this handler's key, False otherwise.
        """

        return key == self.__key

    def _plot(self, plot_data: dict) -> plt.Axes:
        """
        Returns the pre-configured axes regardless of input data.

        This method implements the abstract method from PlotResponsibilityChainBase
        but ignores the input data and simply returns the axes provided at initialization.

        Parameters:
            plot_data (dict): Dictionary containing plot data (ignored).

        Returns:
            plt.Axes: The pre-configured matplotlib Axes object.
        """

        return self.__axes