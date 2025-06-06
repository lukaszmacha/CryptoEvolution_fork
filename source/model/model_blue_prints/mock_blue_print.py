# model/model_blue_prints/mock_blue_print.py

# global imports

# local imports
from source.model import BluePrintBase
from source.model import ModelAdapterBase

class MockBluePrint(BluePrintBase):
    """"""

    def __init__(self, model_to_be_returned: ModelAdapterBase) -> None:
        """"""

        self.__model_adapter_to_be_returned: ModelAdapterBase = model_to_be_returned

    def instantiate_model(self, input_shape: tuple[int, int], output_length: int,
                          spatial_data_shape: tuple[int, int],  **kwargs) -> ModelAdapterBase:
        """
        Returns the pre-configured model regardless of input parameters.

        This method implements the abstract method from BluePrintBase but
        ignores the input parameters and simply returns the model provided
        at initialization.

        Parameters:
            **kwargs: Variable keyword arguments (ignored).

        Returns:
            ModelAdapterBase: The pre-configured model provided at initialization.
        """

        return self.__model_adapter_to_be_returned