# utils/model_blue_print_from_string_converter.py

from typing import Any, Type

from .from_string_converter_base import FromStringConverterBase
from source.model import VGGceptionCnnBluePrint, BluePrintBase, SVMBluePrint, CDT1DCnnBluePrint

class ModelBluePrintFromStringConverter(FromStringConverterBase):
    """
    Converts string identifiers to model blueprint classes.

    This class implements a specific string-to-object conversion for model blueprint
    classes. It maps string names like 'vggception_blueprint' to their corresponding
    blueprint implementations like VGGceptionCnnBluePrint.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the model blueprint converter with blueprint configuration options.

        Parameters:
            **kwargs (dict[str, Any]): Optional parameters that will be passed to
                                      the constructor of the blueprint objects.
        """

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[BluePrintBase]] = {
            'vggception_blueprint': VGGceptionCnnBluePrint,
            'cdt_1d_cnn_blueprint': CDT1DCnnBluePrint,
            'svm_blueprint': SVMBluePrint
        }