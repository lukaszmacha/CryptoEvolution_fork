# model/model_blue_prints/mock_blue_print.py

from tensorflow.keras import Model

from .base_blue_print import BaseBluePrint

class MockBluePrint(BaseBluePrint):
    """
    Mock implementation of the BaseBluePrint for testing purposes.

    This class provides a simple implementation that returns a pre-configured model
    rather than constructing one. It's primarily used in tests to isolate the model
    creation logic from other components being tested.
    """

    def __init__(self, model_to_be_returned: Model) -> None:
        """
        Initialize the mock blueprint with a pre-configured model.

        Parameters:
            model_to_be_returned (Model): Keras model instance that will be returned
                when instantiate_model is called.
        """

        self.__model_to_be_returned: Model = model_to_be_returned

    def instantiate_model(self, **kwargs) -> Model:
        """
        Returns the pre-configured model regardless of input parameters.

        This method implements the abstract method from BaseBluePrint but
        ignores the input parameters and simply returns the model provided
        at initialization.

        Parameters:
            **kwargs: Variable keyword arguments (ignored).

        Returns:
            Model: The pre-configured Keras model provided at initialization.
        """

        return self.__model_to_be_returned