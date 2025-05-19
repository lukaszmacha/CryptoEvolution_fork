# tests/utils/test_model_blue_print_from_string_converter.py

from unittest import TestCase
import logging
from typing import Any
from ddt import ddt, data, unpack

from source.utils import ModelBluePrintFromStringConverter
from source.model import VGGceptionCnnBluePrint

@ddt
class ModelBluePrintFromStringConverterTestCase(TestCase):
    """
    Test case for ModelBluePrintFromStringConverter class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: ModelBluePrintFromStringConverter = ModelBluePrintFromStringConverter()

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
        ({}, 'vggception_blueprint', VGGceptionCnnBluePrint)
    )
    @unpack
    def test_model_blue_print_from_string_converter_convert_from_string(self, params: dict[str, Any], key: str, expected_type: type) -> None:
        """
        Tests ModelBluePrintFromStringConverter's convert_from_string functionality.

        Verifies that convert_from_string correctly creates model blueprint objects
        from string identifiers with the specified configuration parameters.
        Tests the conversion of 'vggception_blueprint' to a VGGceptionCnnBluePrint object.

        Parameters:
            params: Dictionary with parameters for the blueprint configuration.
            key: String identifier of the blueprint type to create.
            expected_type: Expected class type of the created blueprint.

        Asserts:
            The created object is of the expected type.
            All configuration parameters are correctly passed to the blueprint object.
        """

        logging.info(f"Attempt to convert {key} for ModelBluePrintFromStringConverter.")
        self.__update_sut(_kwargs = params)
        result = self.__sut.convert_from_string(key)

        logging.info("Validating expected result.")
        self.assertEqual(type(result), expected_type)
        for param_name, expected_param_value in params.items():
            param_value = getattr(result, param_name, None)
            self.assertEqual(param_value, expected_param_value)
