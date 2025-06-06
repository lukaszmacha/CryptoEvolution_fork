# utils/label_annotator_from_string_converter.py

# global imports
from typing import Any, Type

# local imports
from source.utils import FromStringConverterBase
from source.environment import LabelAnnotatorBase, SimpleLabelAnnotator

class LabelAnnotatorFromStringConverter(FromStringConverterBase):
    """"""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """"""

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[LabelAnnotatorBase]] = {
            'simple': SimpleLabelAnnotator
        }