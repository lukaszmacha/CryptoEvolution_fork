# agent/strategies/classification_testing_strategy_handler.py

# global imports
from typing import Any
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# local imports
from source.environment import TradingEnvironment
from source.agent import TestingStrategyHandlerBase
from source.agent import ClassificationTestable

class ClassificationTestingStrategyHandler(TestingStrategyHandlerBase):
    """"""

    PLOTTING_KEY: str = 'classification_testing'

    def evaluate(self, testable_agent: ClassificationTestable, environment: TradingEnvironment) -> \
        tuple[list[str], list[dict[str, Any]]]:
        """"""

        classes = list(environment.get_trading_consts().OUTPUT_CLASSES.keys())
        input_data, output_data = environment.get_labeled_data()

        # input_data = np.squeeze(input_data, axis=1)
        # output_data = np.argmax(output_data, axis=1)
        prediction_probabilities = testable_agent.classify(input_data)
        print(prediction_probabilities)

        y_true = np.argmax(output_data, axis=1)
        y_pred = np.argmax(prediction_probabilities, axis=1)

        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, target_names = classes,
                                             output_dict = True, zero_division = 0)

        summary = {
            "output_data": output_data,
            "prediction_probabilities": prediction_probabilities,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
            "accuracy": (y_true == y_pred).mean()
        }

        return [ClassificationTestingStrategyHandler.PLOTTING_KEY], [summary]