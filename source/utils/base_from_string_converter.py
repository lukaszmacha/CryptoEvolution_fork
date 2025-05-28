# utils/base_from_string_converter.py

from typing import Any
import logging

class BaseFromStringConverter():
    """
    Base class for string-to-object conversion functionality.

    This class provides a framework for converting string identifiers into actual
    object instances. Subclasses implement specific conversion logic by defining
    value maps that link string identifiers to their corresponding classes or values.
    All converter implementations should inherit from this class and override
    the __init__ method.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the converter with optional parameters.

        Parameters:
            **kwargs (dict[str, Any]): Optional parameters that will be passed to
                                      the constructor of the converted objects.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """

        raise NotImplementedError("Subclasses must implement this method")

    def convert_from_string(self, string_value: str) -> Any:
        """
        Converts a string identifier to its corresponding object.

        Uses the _value_map dictionary from the subclass to convert the string
        to an actual object instance by looking up the corresponding class and
        instantiating it with the parameters provided at initialization.

        Parameters:
            string_value (str): String identifier to convert.

        Returns:
            Any: An instance of the class corresponding to the string identifier.
                 Returns None if conversion fails.
        """

        converted_value = self._value_map.get(string_value, None)
        if converted_value is None:
            logging.warning(f'Did not managed to convert {string_value}, returning None.')

        return converted_value(**self._kwargs)