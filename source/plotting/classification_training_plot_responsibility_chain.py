# plotting/classification_training_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# local imports
from source.agent import ClassificationLearningStrategyHandler
from source.plotting import PlotResponsibilityChainBase

class ClassificationTrainingPlotResponsibilityChain(PlotResponsibilityChainBase):
    """"""

    def _can_plot(self, key: str) -> bool:
        """"""

        return key == ClassificationLearningStrategyHandler.PLOTTING_KEY

    def _plot(self, plot_data: dict) -> plt.Axes:
        """"""

        history = plot_data.get("history", None)
        currency_prices = plot_data.get("currency_prices", None)
        if history is not None :
            loss = history.get("loss", None)
            accuracy = history.get("accuracy", None)

        if loss is None or accuracy is None or currency_prices is None:
            logging.warning("Insufficient data for plotting classification results.")
            plt.text(0.5, 0.5, "Insufficient data for plotting",
                     ha = 'center', va = 'center', fontsize = 12)
            return plt.gca()

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(2, 1, figure = fig)

        # Plot 1: Training loss and accuracy
        plt.subplot(gs[0, 0])
        plt.title("Training loss and accuracy")
        plt.plot(loss, label = 'Loss')
        plt.plot(accuracy, label = 'Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend(loc = 'upper left')

        # Plot 2: Close price
        plt.subplot(gs[1, 0])
        plt.title("Close Price, granularity: 1 day")
        plt.plot(currency_prices, label = 'Close Price')
        plt.xlabel('Trading steps')
        plt.ylabel('Price')
        plt.legend(loc = 'upper left')
        plt.tight_layout()

        return plt.gca()
