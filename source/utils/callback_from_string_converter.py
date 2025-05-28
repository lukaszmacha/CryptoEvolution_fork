# utils/callback_from_string_converter.py

from typing import Any, Type
from  tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping

from .base_from_string_converter import BaseFromStringConverter

class CallbackFromStringConverter(BaseFromStringConverter):
    """
    Converts string identifiers to Keras callback classes.

    This class implements a specific string-to-object conversion for Keras callback
    classes. It maps string names like 'reduce_rl_on_plateau' to their corresponding
    callback implementations like ReduceLROnPlateau.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the callback converter with options for callback configuration.

        Parameters:
            **kwargs (dict[str, Any]): Optional parameters that will be passed to
                                      the constructor of the callback objects.
        """

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[Callback]] = {
            'reduce_rl_on_plateau': ReduceLROnPlateau,
            'early_stopping': EarlyStopping
        }