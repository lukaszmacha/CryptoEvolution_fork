# model/model_blue_prints/base_blue_print.py

from tensorflow.keras import Model

class BaseBluePrint():
    """
    Base class for model blue prints.

    This class serves as an abstract base for different neural network architecture
    blueprints used in the application. All model blueprint implementations should
    inherit from this class and implement the instantiate_model method.
    """

    def instantiate_model(self, *args) -> Model:
        """
        Creates and returns a Keras model instance based on the blueprint.

        This is an abstract method that should be implemented by all subclasses.
        The implementation should construct a neural network architecture and
        return it as a Keras Model object.

        Parameters:
            *args: Variable length argument list for model configuration.

        Returns:
            Model: A Keras model to be compiled further.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """

        raise NotImplementedError("Subclasses must implement this method")