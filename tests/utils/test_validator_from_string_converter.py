# tests/utils/test_validator_from_string_converter.py

from unittest import TestCase
import logging
from typing import Any
from ddt import ddt, data, unpack

from source.utils import ValidatorFromStringConverter
from source.environment import PointsRewardValidator, PriceRewardValidator

@ddt
class ValidatorFromStringConverterTestCase(TestCase):
    """
    Test case for ValidatorFromStringConverter class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: ValidatorFromStringConverter = ValidatorFromStringConverter()

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
            'coefficient': 2.0,
            'normalizable': True
        }, 'price_reward_validator', PriceRewardValidator),
        ({
            'rewarded_points': (3, -1),
        }, 'points_reward_validator', PointsRewardValidator)
    )
    @unpack
    def test_validator_from_string_converter_convert_from_string(self, params: dict[str, Any], key: str, expected_type: type) -> None:
        """
        Tests ValidatorFromStringConverter's convert_from_string functionality.

        Verifies that convert_from_string correctly creates reward validator objects
        from string identifiers with the specified configuration parameters.

        Parameters:
            params: Dictionary with parameters for the validator configuration.
            key: String identifier of the validator type to create.
            expected_type: Expected class type of the created validator.

        Asserts:
            The created object is of the expected type.
            All configuration parameters are correctly passed to the validator object.
        """

        logging.info(f"Attempt to convert {key} for ValidatorFromStringConverter.")
        self.__update_sut(_kwargs = params)
        result = self.__sut.convert_from_string(key)

        logging.info("Validating expected result.")
        self.assertEqual(type(result), expected_type)
        for param_name, expected_param_value in params.items():
            class_name = result.__class__.__name__
            param_value = getattr(result, f'_{class_name}__{param_name}', None)
            self.assertEqual(param_value, expected_param_value)
