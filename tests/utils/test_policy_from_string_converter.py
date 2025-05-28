# tests/utils/test_policy_from_string_converter.py

from unittest import TestCase
import logging
from typing import Any
from ddt import ddt, data, unpack
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy

from source.utils import PolicyFromStringConverter

@ddt
class PolicyFromStringConverterTestCase(TestCase):
    """
    Test case for PolicyFromStringConverter class. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        self.__sut: PolicyFromStringConverter = PolicyFromStringConverter()

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
            'tau': 0.5
        }, 'boltzmann', BoltzmannQPolicy),
        ({
            'eps': 1e-3
        }, 'eps_greedy', EpsGreedyQPolicy),
        ({
            'attr': 'tau',
            'value_max': 0.5,
            'value_min': 0.1,
            'value_test': 0.1,
            'nb_steps': 1000
        }, 'linear_annealed_boltzmann', LinearAnnealedPolicy),
        ({
            'attr': 'eps',
            'value_max': 0.05,
            'value_min': 0.01,
            'value_test': 0.01,
            'nb_steps': 1000
        }, 'linear_annealed_eps_greedy', LinearAnnealedPolicy)
    )
    @unpack
    def test_policy_from_string_converter_convert_from_string(self, params: dict[str, Any], key: str, expected_type: type) -> None:
        """
        Tests PolicyFromStringConverter's convert_from_string functionality.

        Verifies that convert_from_string correctly creates RL policy objects
        from string identifiers with the specified configuration parameters.

        Parameters:
            params: Dictionary with parameters for the policy configuration.
            key: String identifier of the policy type to create.
            expected_type: Expected class type of the created policy.

        Asserts:
            The created object is of the expected type.
            All configuration parameters are correctly passed to the policy object.
        """

        logging.info(f"Attempt to convert {key} for PolicyFromStringConverter.")
        self.__update_sut(_kwargs = params)
        result = self.__sut.convert_from_string(key)

        logging.info("Validating expected result.")
        self.assertEqual(type(result), expected_type)
        for param_name, expected_param_value in params.items():
            param_value = getattr(result, param_name, None)
            self.assertEqual(param_value, expected_param_value)
