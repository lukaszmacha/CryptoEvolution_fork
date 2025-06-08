# plotting/summary_plot_responsibility_chain.py

# global imports
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# local imports
from source.plotting import PlotResponsibilityChainBase

class SummaryPlotResponsibilityChain(PlotResponsibilityChainBase):
    """"""

    def _can_plot(self, key: str) -> bool:
        """"""

        self.__key = key
        return key == 'summary' or key == 'volatility'

    def _plot(self, plot_data: dict) -> plt.Axes:
        """"""

        train = plot_data.get("train", None)
        test = plot_data.get("test", None)

        fig = plt.figure(figsize = self._EXPECTED_FIGURE_SIZE)
        gs = GridSpec(2, 1, figure = fig)

        if self.__key == 'summary':
            # Plot 1: Currency prices
            plt.subplot(gs[0, 0])
            if train is None or test is None:
                logging.warning("Insufficient data for plotting summary results.")
                plt.text(0.5, 0.5, "Insufficient data for plotting",
                        ha = 'center', va = 'center', fontsize = 12)
            else:
                plt.title("Currency Prices")

                # Plot training data
                train_x = range(len(train['price']))
                plt.plot(train_x, train['price'], label='Train Prices')

                # Plot test data, starting from where train ends
                test_x = range(len(train['price']), len(train['price']) + len(test['price']))
                plt.plot(test_x, test['price'], label='Test Prices')

                # Add a vertical line to mark the train/test split
                plt.axvline(x=len(train['price'])-1, color='r', linestyle='--', alpha=0.7, label='Train/Test Split')

                plt.yscale('log')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend(loc='upper left')


            # Plot 2: Labels
            plt.subplot(gs[1, 0])
            if train is None or test is None:
                logging.warning("Insufficient data for plotting summary results.")
                plt.text(0.5, 0.5, "Insufficient data for plotting",
                        ha='center', va='center', fontsize=12)
            else:
                plt.title("Price Changes by Trend Classification")

                # Define colors for each class (UP, DOWN, NO_TREND)
                trend_colors = ['green', 'red', 'gray']  # Green for UP, Red for DOWN, Gray for NO_TREND

                # Make sure we have trend classification data
                if 'labels' in train and 'labels' in test:
                    logging.info(f"{train['labels'].shape}, {test['labels'].shape}")
                    logging.info(f"{len(train_x)}, {len(test_x)}")
                    train_x = range(len(train['labels']))
                    test_x = range(len(train['labels']), len(train['labels']) + len(test['labels']))
                    # Plot training data segments colored by classification
                    for i in range(1, len(train_x)):
                        class_idx = np.argmax(train['labels'][i])
                        color = trend_colors[class_idx]
                        plt.plot([train_x[i-1], train_x[i]],
                                [train['price'][i-1], train['price'][i]],
                                color=color, linewidth=1.5)

                    # Plot test data segments colored by classification
                    for i in range(1, len(test_x)):
                        class_idx = np.argmax(test['labels'][i])
                        color = trend_colors[class_idx]
                        plt.plot([test_x[i-1], test_x[i]],
                                [test['price'][i-1], test['price'][i]],
                                color=color, linewidth=1.5)

                    # Add a custom legend for the trend classes
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='green', lw=2, label='UP Trend'),
                        Line2D([0], [0], color='red', lw=2, label='DOWN Trend'),
                        Line2D([0], [0], color='gray', lw=2, label='NO Trend')
                    ]
                    plt.legend(handles=legend_elements, loc='upper left')

                plt.yscale('log')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.title('Price Changes by Trend Classification')
        else:
            # Plot 1: Volatility
            plt.subplot(gs[0, 0])
            if train is None or test is None:
                logging.warning("Insufficient data for plotting summary results.")
                plt.text(0.5, 0.5, "Insufficient data for plotting",
                        ha='center', va='center', fontsize=12)
            else:
                plt.title("Volatility")

                # Plot training data
                train_x = range(len(train['price']))

                # Plot test data, starting from where train ends
                test_x = range(len(train['price']), len(train['price']) + len(test['price']))

                # Plot training data
                plt.plot(train_x, train['volatility'], label='Train Volatility')

                # Plot test data, starting from where train ends
                plt.plot(test_x, test['volatility'], label='Test Volatility')

                plt.xlabel('Time')
                plt.ylabel('Volatility')
                plt.legend(loc='upper left')

            # Plot 2: Price changes with volatility
            plt.subplot(gs[1, 0])
            if train is None or test is None:
                logging.warning("Insufficient data for plotting summary results.")
                plt.text(0.5, 0.5, "Insufficient data for plotting",
                        ha='center', va='center', fontsize=12)
            else:
                plt.title("Price Changes with Volatility")

                # Create a colormap - 'plasma' works well for visualizing intensity
                from matplotlib.cm import get_cmap
                colormap = get_cmap('hot')

                # For train data: Plot line segments colored by volatility
                # Normalize volatility to 0-1 range for color mapping
                train_vol_normalized = np.array(train['volatility']) / max(train['volatility'])

                # Plot colored line segments for training data
                for i in range(1, len(train_x)):
                    plt.plot([train_x[i-1], train_x[i]],
                            [train['price'][i-1], train['price'][i]],
                            color=colormap(train_vol_normalized[i]),
                            linewidth=1.5)

                # For test data: Plot line segments colored by volatility
                test_vol_normalized = np.array(test['volatility']) / max(test['volatility'])

                # Plot colored line segments for test data
                for i in range(1, len(test_x)):
                    plt.plot([test_x[i-1], test_x[i]],
                            [test['price'][i-1], test['price'][i]],
                            color=colormap(test_vol_normalized[i]),
                            linewidth=1.5)

                # Add a color bar to show volatility scale
                import matplotlib.cm as cm
                import matplotlib.colors as colors

                # Create a scalar mappable for the colorbar
                norm = colors.Normalize(vmin=0, vmax=max(np.array(train['volatility']).max(), np.array(test['volatility']).max()))
                sm = cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])

                # Add colorbar
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_label('Volatility')

                plt.yscale('log')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.title('Price with Volatility-Colored Line')

        plt.tight_layout()

        return plt.gca()
