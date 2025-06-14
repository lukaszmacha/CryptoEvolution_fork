# model/model_adapters/tf_model_adapter.py

# global imports
from tensorflow.keras.models import Model
from typing import Callable
from tensorflow.keras.optimizers import Optimizer
from typing import Any, Optional
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical

# local imports
from source.model import ModelAdapterBase

class TFModelAdapter(ModelAdapterBase):
    """"""

    WEIGHTS_FILE_EXTENSION: str = ".h5"
    OPTIMIZER = "adam"
    LOSS = "categorical_crossentropy"
    METRICS = ["accuracy"]

    def __init__(self, model: Model, optimizer: Optional[Optimizer] = None,
                 loss: Optional[str] = None, metrics: Optional[list[str]] = None) -> None:
        """"""

        if optimizer is None:
            optimizer = TFModelAdapter.OPTIMIZER
        if loss is None:
            loss = TFModelAdapter.LOSS
        if metrics is None:
            metrics = TFModelAdapter.METRICS

        self.__model: Model = model
        self.__model.compile(optimizer = optimizer,
                             loss = loss,
                             metrics = metrics)

    def load_model(self, path: str) -> None:
        """"""

        if TFModelAdapter.WEIGHTS_FILE_EXTENSION not in path:
            raise ValueError(f"Model path must end with '{TFModelAdapter.WEIGHTS_FILE_EXTENSION}'.")

        self.__model.load_weights(path)

    def save_model(self, path: str) -> None:
        """"""

        if TFModelAdapter.WEIGHTS_FILE_EXTENSION not in path:
            raise ValueError(f"Model path must end with '{TFModelAdapter.WEIGHTS_FILE_EXTENSION}'.")

        self.__model.save_weights(path)

    def print_summary(self, print_function: Callable = print) -> None:
        """"""

        self.__model.summary(print_fn = print_function)

    def fit(self, input_data: Any, output_data: Any, **kwargs) -> dict:
        """"""

        X_train, X_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.1,
                                                          random_state=42, stratify=output_data)

        X_train = np.squeeze(X_train, axis=1)
        y_train = np.argmax(y_train, axis=1)

        from imblearn.over_sampling import ADASYN
        sampling_strategy = {0: 10000, 1: 10000, 2: 10000}
        X_train, y_train = ADASYN(sampling_strategy=sampling_strategy).fit_resample(X_train, y_train)
        X_train = np.expand_dims(np.array(X_train), axis=1)
        y_train = to_categorical(np.array(y_train), num_classes=3)

        return self.__model.fit(X_train, y_train,
                                validation_data=(X_val, y_val), **kwargs).history

    def predict(self, data: Any) -> dict:
        """"""

        return self.__model.predict(data)

    def get_model(self) -> Model:
        """"""

        return self.__model