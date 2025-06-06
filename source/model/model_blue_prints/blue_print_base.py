# model/model_blue_prints/blue_print_base.py

# global imports

# local imports
from source.model import ModelAdapterBase

class BluePrintBase():
    """"""

    def instantiate_model(self, input_shape: tuple[int, int], output_length: int,
                          spatial_data_shape: tuple[int, int], *args) -> ModelAdapterBase:
        """"""

        raise NotImplementedError("Subclasses must implement this method")