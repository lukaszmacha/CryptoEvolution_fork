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
        if history is not None :
            loss = history.get("loss", None)
            val_loss = history.get("val_loss", None)
            accuracy = history.get("accuracy", None)
            val_accuracy = history.get("val_accuracy", None)
        else:
            loss = None
            accuracy = None

        learning_curve_data = plot_data.get("learning_curve_data", None)

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)

        # Plot 1: Training loss and accuracy
        if loss is not None and accuracy is not None:
            plt.title("Training loss and accuracy")
            plt.plot(loss, label = 'Loss')
            plt.plot(accuracy, label = 'Accuracy')
            plt.plot(val_loss, label = 'Validation Loss')
            plt.plot(val_accuracy, label = 'Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend(loc = 'upper left')
        elif learning_curve_data is not None:
            plt.title("Learning Curve")

            # Plot training score with error bands
            plt.fill_between(learning_curve_data["train_sizes"],
                            learning_curve_data["train_scores_mean"] - learning_curve_data["train_scores_std"],
                            learning_curve_data["train_scores_mean"] + learning_curve_data["train_scores_std"],
                            alpha=0.1, color="r")
            plt.plot(learning_curve_data["train_sizes"], learning_curve_data["train_scores_mean"],
                    'o-', color="r", label="Training score")

            # Plot cross-validation score with error bands
            plt.fill_between(learning_curve_data["train_sizes"],
                            learning_curve_data["valid_scores_mean"] - learning_curve_data["valid_scores_std"],
                            learning_curve_data["valid_scores_mean"] + learning_curve_data["valid_scores_std"],
                            alpha=0.1, color="g")
            plt.plot(learning_curve_data["train_sizes"], learning_curve_data["valid_scores_mean"],
                    'o-', color="g", label="Cross-validation score")

            plt.grid()
            plt.xlabel('Training examples')
            plt.ylabel('Accuracy')
            plt.legend(loc="best")

        plt.tight_layout()

        return plt.gca()
