# tests/utils/test_callback_from_string_converter.py

from unittest import TestCase
import logging
from typing import Any
from ddt import ddt, data, unpack
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from source.utils import CallbackFromStringConverter

@ddt
class CallbackFromStringConverterTestCase(TestCase):
    """
    Test case for CallbackFromStringConverter class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: CallbackFromStringConverter = CallbackFromStringConverter()

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")

    def __update_sut(self, **kwargs) -> None:
        """
        Allows to update already created sut. It speeds up test
        cases' scenarios by enabling injections of certain values
        also into private sut members.
        """

        for name, value in kwargs.items():
            for attribute_name in self.__sut.__dict__:
                if name in attribute_name:
                    setattr(self.__sut, attribute_name, value)

    @data(
        ({
            'monitor': 'accuracy',
            'patience': 10,
            'verbose': 1,
            'restore_best_weights': True
        }, 'early_stopping', EarlyStopping),
        ({
            'monitor': 'accuracy',
            'patience': 5,
            'verbose': 2,
            'min_lr': 0.0005,
            'factor': 0.5,
        }, 'reduce_rl_on_plateau', ReduceLROnPlateau)
    )
    @unpack
    def test_callback_from_string_converter_convert_from_string(self, params: dict[str, Any], key: str, expected_type: type) -> None:
        """
        Tests CallbackFromStringConverter's convert_from_string functionality.

        Verifies that convert_from_string correctly creates Keras callback objects
        from string identifiers with the specified configuration parameters.

        Parameters:
            params: Dictionary with parameters for the callback configuration.
            key: String identifier of the callback type to create.
            expected_type: Expected class type of the created callback.

        Asserts:
            The created object is of the expected type.
            All configuration parameters are correctly passed to the callback object.
        """

        logging.info(f"Attempt to convert {key} for CallbackFromStringConverter.")
        self.__update_sut(_kwargs = params)
        result = self.__sut.convert_from_string(key)

        logging.info("Validating expected result.")
        self.assertEqual(type(result), expected_type)
        for param_name, expected_param_value in params.items():
            param_value = getattr(result, param_name, None)
            self.assertEqual(param_value, expected_param_value)
