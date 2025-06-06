# utils/policy_from_string_converter.py

from typing import Any, Type
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, Policy

from .from_string_converter_base import FromStringConverterBase

class PolicyFromStringConverter(FromStringConverterBase):
    """
    Converts string identifiers to reinforcement learning policy classes.

    This class implements a specific string-to-object conversion for RL policy
    classes. It maps string names to their corresponding policy implementations,
    with special handling for linear annealed policies that require wrapping
    the base policy.
    """

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """
        Initializes the policy converter with options for policy configuration.

        Parameters:
            **kwargs (dict[str, Any]): Optional parameters that will be passed to
                                      the constructor of the policy objects.
        """

        self._kwargs: dict[str, Any] = kwargs
        self._value_map: dict[str, Type[Policy]] = {
            'boltzmann': BoltzmannQPolicy,
            'eps_greedy': EpsGreedyQPolicy,
            'linear_annealed_boltzmann': BoltzmannQPolicy,
            'linear_annealed_eps_greedy': EpsGreedyQPolicy
        }

    def convert_from_string(self, string_value: str) -> Policy:
        """
        Converts a string identifier to its corresponding policy object.

        Overrides the base method to provide special handling for linear annealed
        policies, which require wrapping the base policy in a LinearAnnealedPolicy.

        Parameters:
            string_value (str): String identifier of the policy.

        Returns:
            Policy: An instance of the appropriate policy class.
                   For linear annealed policies, returns the base policy wrapped
                   in a LinearAnnealedPolicy. Otherwise, returns the base policy.
        """

        if 'linear_annealed' in string_value and string_value in self._value_map:
            converted_value = self._value_map.get(string_value)
            return LinearAnnealedPolicy(converted_value(), **self._kwargs)
        else:
            return super().convert_from_string(string_value)