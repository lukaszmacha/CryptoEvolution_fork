# utils/optimizer_from_string_converter.py

from typing import Any, Type
from tensorflow.keras.optimizers import Adam, Optimizer

from .from_string_converter_base import FromStringConverterBase

class OptimizerFromStringConverter(FromStringConverterBase):
    """
    Converts string identifiers to Keras optimizer classes.

    This class implements a specific string-to-object conversion for Keras optimizer
    classes. It maps string names like 'adam' to their corresponding optimizer
    implementations.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the optimizer converter with options for optimizer configuration.

        Parameters:
            **kwargs (dict[str, Any]): Optional parameters that will be passed to
                                      the constructor of the optimizer objects.
        """

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[Optimizer]] = {
            'adam': Adam
        }