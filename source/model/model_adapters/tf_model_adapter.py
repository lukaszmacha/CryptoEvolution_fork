# model/model_adapters/tf_model_adapter.py

# global imports
from tensorflow.keras.models import Model
from typing import Callable
from tensorflow.keras.optimizers import Optimizer
from typing import Any, Optional

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

        return self.__model.fit(input_data, output_data,
                                validation_split = 0.1, **kwargs)

    def predict(self, data: Any) -> dict:
        """"""

        return self.__model.predict(data)

    def get_model(self) -> Model:
        """"""

        return self.__model