# utils/validator_from_string_converter.py

from typing import Any, Type

from .base_from_string_converter import BaseFromStringConverter
from ..environment.points_reward_validator import PointsRewardValidator
from ..environment.price_reward_validator import PriceRewardValidator
from ..environment.reward_validator_base import RewardValidatorBase

class ValidatorFromStringConverter(BaseFromStringConverter):
    """
    Converts string identifiers to reward validator classes.

    This class implements a specific string-to-object conversion for reward validator
    classes. It maps string names like 'price_reward_validator' to their corresponding
    validator implementations.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the validator converter with options for validator configuration.

        Parameters:
            **kwargs (dict[str, Any]): Optional parameters that will be passed to
                                      the constructor of the validator objects.
        """

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[RewardValidatorBase]] = {
            'price_reward_validator': PriceRewardValidator,
            'points_reward_validator': PointsRewardValidator
        }