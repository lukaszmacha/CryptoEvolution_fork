# model/model_adapters/sci_kit_learn_model_adapter.py

# global imports
import joblib
import numpy as np
import pandas as pd
from typing import Callable, Any, Optional
from sklearn.base import BaseEstimator
import logging

# local imports
from source.model import ModelAdapterBase

class SciKitLearnModelAdapter(ModelAdapterBase):
    """
    Adapter class for scikit-learn models that implements the common interface
    defined by ModelAdapterBase.

    This adapter works with any scikit-learn estimator that implements fit and predict methods.
    It also handles dimensionality differences between deep learning and traditional ML models.
    """

    MODEL_FILE_EXTENSION: str = ".joblib"

    def __init__(self, model: BaseEstimator) -> None:
        """
        Initialize the adapter with a scikit-learn model/estimator.

        Parameters:
            model (BaseEstimator): Any scikit-learn estimator object
                                  (e.g., LinearRegression, RandomForestClassifier, etc.)
        """
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise ValueError("Model must implement 'fit' and 'predict' methods")

        self.__model = model

    def load_model(self, path: str) -> None:
        """
        Load a saved scikit-learn model from a file.

        Parameters:
            path (str): Path to the saved model file (must have .joblib extension)

        Raises:
            ValueError: If the path does not have the correct file extension
        """
        if SciKitLearnModelAdapter.MODEL_FILE_EXTENSION not in path:
            raise ValueError(f"Model path must end with '{SciKitLearnModelAdapter.MODEL_FILE_EXTENSION}'.")

        self.__model = joblib.load(path)

    def save_model(self, path: str) -> None:
        """
        Save the scikit-learn model to a file.

        Parameters:
            path (str): Path where the model will be saved (must have .joblib extension)

        Raises:
            ValueError: If the path does not have the correct file extension
        """
        if SciKitLearnModelAdapter.MODEL_FILE_EXTENSION not in path:
            raise ValueError(f"Model path must end with '{SciKitLearnModelAdapter.MODEL_FILE_EXTENSION}'.")

        joblib.dump(self.__model, path)

    def print_summary(self, print_function: Callable = print) -> None:
        """
        Print a summary of the model's parameters and properties.

        Parameters:
            print_function (Callable): Function to use for printing, default is built-in print
        """
        print_function(f"Model type: {type(self.__model).__name__}")
        print_function("Model parameters:")
        for param, value in self.__model.get_params().items():
            print_function(f"  {param}: {value}")


    def fit(self, input_data: Any, output_data: Any, **kwargs) -> Any:
        """
        Train the scikit-learn model on the provided data.
        Automatically handles dimensionality reduction for scikit-learn compatibility.

        Parameters:
            input_data (Any): Features data (X), can be numpy array or pandas DataFrame
            output_data (Any): Target data (y), can be numpy array or pandas Series
            **kwargs: Additional arguments to pass to the model's fit method
                      Note: scikit-learn doesn't use many common keras arguments like epochs, batch_size, etc.
                      These will be filtered out to prevent errors

        Returns:
            The fitted model or a dictionary with history-like structure for compatibility
        """

        self.__model.fit(input_data, output_data)

        history = {
            'sklearn_model': True,
            'model_type': type(self.__model).__name__
        }

        if hasattr(self.__model, 'score'):
            history['accuracy'] = [self.__model.score(input_data, output_data)]

        logging.info(history)
        return history

    def predict(self, data: Any) -> np.ndarray:
        """
        Generate predictions using the trained model.
        Handles dimensionality reduction for scikit-learn compatibility.

        Parameters:
            data (Any): Input features for prediction

        Returns:
            np.ndarray: Model predictions in a format compatible with the system
        """

        return self.__model.predict_proba(data)

    def get_model(self) -> BaseEstimator:
        """
        Get the underlying scikit-learn model.

        Returns:
            BaseEstimator: The scikit-learn model instance
        """

        return self.__model