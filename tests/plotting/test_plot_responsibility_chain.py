# tests/plotting/test_plot_responsibility_chain.py

from unittest import TestCase
import logging
import matplotlib.pyplot as plt
from ddt import ddt, data, unpack

from mock_plot_responsibility_chain import MockPlotResponsibilityChain

@ddt
class PlotResponsibilityChainTestCase(TestCase):
    """
    Test case for PlotResponsibilityChain pattern implementation. Stores all the test cases
    and allows for convenient test case execution.
    """

    def setUp(self) -> None:
        """
        Setup function responsible for creation of system under
        test (sut) for this class.
        """

        logging.info("Setting up test environment.")
        first_chain_link = MockPlotResponsibilityChain(plt.subplots()[1], 'key_1')
        second_chain_link = MockPlotResponsibilityChain(plt.subplots()[1], 'key_2')
        third_chain_link = MockPlotResponsibilityChain(plt.subplots()[1], 'key_3')

        second_chain_link.add_next_chain_link(third_chain_link)
        first_chain_link.add_next_chain_link(second_chain_link)

        self.__sut: MockPlotResponsibilityChain = first_chain_link

    def tearDown(self) -> None:
        """
        Tear down function responsible for cleaning up all the
        needed dependencies between test cases.
        """

        logging.info("Tearing down test environment.")
        plt.close('all')

    @data(
        ('key_1', plt.Axes),
        ('key_2', plt.Axes),
        ('key_3', plt.Axes),
        ('key_4', type(None))
    )
    @unpack
    def test_plot_responsibility_chain_plot(self, key: str, expected_result_type: type) -> None:
        """
        Tests the responsibility chain pattern implementation for plot handling.

        Verifies that the plot method correctly handles requests by finding the
        appropriate handler in the chain based on the provided key. Tests that
        requests with matching keys are processed by the corresponding handler
        and that unrecognized keys result in None being returned.

        Parameters:
            key (str): The key to test with the responsibility chain.
            expected_result_type (type): Expected return type based on the key.

        Asserts:
            The result is of the expected type (plt.Axes for recognized keys,
            None for unrecognized keys).
        """

        logging.info("Starting plot test.")
        mocked_input_data = {
            'key': key,
            'plot_data': []
        }

        logging.info("Plotting using provided data.")
        result = self.__sut.plot(mocked_input_data)

        logging.info("Checking expected result type.")
        self.assertTrue(isinstance(result, expected_result_type))
