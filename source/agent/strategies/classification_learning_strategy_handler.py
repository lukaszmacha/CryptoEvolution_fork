# agent/strategies/classification_learning_strategy_handler.py

# global imports
from typing import Any
from tensorflow.keras.callbacks import Callback
import logging
import numpy as np
from sklearn.model_selection import learning_curve
from tensorflow.keras.utils import to_categorical

# local imports
from source.agent import LearningStrategyHandlerBase
from source.agent import AgentBase
from source.agent import ClassificationLearningAgent
from source.environment import TradingEnvironment
from source.model import BluePrintBase
from source.model import TFModelAdapter
from source.model import SciKitLearnModelAdapter

class ClassificationLearningStrategyHandler(LearningStrategyHandlerBase):
    """"""

    # global constants
    PLOTTING_KEY: str = 'classification_learning'

    def create_agent(self, model_blue_print: BluePrintBase,
                     trading_environment: TradingEnvironment) -> AgentBase:
        """"""

        windows_size = trading_environment.get_trading_consts().WINDOW_SIZE
        spatial_data_shape = trading_environment.get_environment_spatial_data_dimension()
        market_data_shape = (spatial_data_shape[1] * windows_size, )

        number_of_classes = len(trading_environment.get_trading_consts().OUTPUT_CLASSES)
        model_adapter = model_blue_print.instantiate_model(market_data_shape, number_of_classes,
                                                           spatial_data_shape)
        return ClassificationLearningAgent(model_adapter)

    def fit(self, agent: ClassificationLearningAgent, environment: TradingEnvironment,
            nr_of_steps: int, nr_of_episodes: int, callbacks: list[Callback]) -> tuple[list[str], list[dict[str, Any]]]:
        """"""

        if not isinstance(agent, ClassificationLearningAgent):
            raise TypeError("Agent must be an instance of ClassificationLearningAgent.")

        input_data, output_data = environment.get_labeled_data()
        steps_per_epoch = nr_of_steps // nr_of_episodes
        batch_size = len(input_data) // steps_per_epoch
        if batch_size <= 0:
            logging.warning("Batch size is zero or negative, using value of 1 instead.")
            batch_size = 1

        env_length = environment.get_environment_length()
        currency_prices = environment.get_data_for_iteration(['close'], 0, env_length - 1)
        currency_prices = (np.array(currency_prices) / currency_prices[0]).tolist()
        tensorflow_arguments = {}
        learning_curve_data = {}

        if isinstance(agent._model_adapter, TFModelAdapter):
            logging.info("Using TensorFlow model adapter for training.")
            tensorflow_arguments = {
                "batch_size": batch_size,
                "epochs": nr_of_episodes,
                "callbacks": callbacks
            }
            from imblearn.over_sampling import ADASYN
            input_data = np.squeeze(input_data, axis=1)
            output_data = np.argmax(output_data, axis=1)
            sampling_strategy = {0: 10000, 1: 10000, 2: 10000}
            input_data, output_data = ADASYN(sampling_strategy=sampling_strategy).fit_resample(input_data, output_data)
            input_data = np.expand_dims(np.array(input_data), axis = 1)
            nr_of_classes = len(environment.get_trading_consts().OUTPUT_CLASSES)
            output_data = to_categorical(np.array(output_data), num_classes = nr_of_classes)

        logging.info(f"{np.sum(output_data, axis = 0)}")
        logging.info(f"Input data shape: {input_data.shape}, Output data shape: {output_data.shape}")

        if isinstance(agent._model_adapter, SciKitLearnModelAdapter):
            logging.info("Using SciKitLearn model adapter for training.")
            # Get the underlying model
            model = agent._model_adapter.get_model()

            # Generate learning curve data
            train_sizes = np.linspace(0.1, 1.0, 5)  # 5 points for faster computation
            train_sizes_abs, train_scores, valid_scores = learning_curve(
                model, input_data, output_data,
                train_sizes=train_sizes, cv=5,
                scoring='accuracy', n_jobs=-1
            )

            # Store learning curve data
            learning_curve_data = {
                "train_sizes": train_sizes_abs,
                "train_scores_mean": np.mean(train_scores, axis=1),
                "train_scores_std": np.std(train_scores, axis=1),
                "valid_scores_mean": np.mean(valid_scores, axis=1),
                "valid_scores_std": np.std(valid_scores, axis=1)
            }

        return [ClassificationLearningStrategyHandler.PLOTTING_KEY], \
            [{"history": agent.classification_fit(input_data, output_data, **tensorflow_arguments),
              "currency_prices": currency_prices,
              "learning_curve_data": learning_curve_data}]
