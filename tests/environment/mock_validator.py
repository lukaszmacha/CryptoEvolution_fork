# tests/environment/mock_validator.py

from typing import Callable

from source.environment import RewardValidatorBase, Order

class MockRewardValidator(RewardValidatorBase):
    """
    Implements validator that validation policy can be specified from outside of the class.
    This allows for creation of simple validators to be mocked during testing.
    """

    def __init__(self, mocking_fucntion: Callable[[list[Order]], float]) -> None:
        """
        Class constructor.

        Parameters:
            mocking_fucntion (Callable[[list[Order]], float]): Allows to specify
                validator's policy from outside of the class.
        """

        self.lambda_mocking_function: Callable[[list[Order]], float] = mocking_fucntion

    def validate_orders(self, orders: list[Order]) -> float:
        """
        Calculates number of points to be rewarded for list of closed trades.

        Parameters:
            orders (list[Order]): Orders to be validated.

        Returns:
            (float): Calcualted reward.
        """

        return self.lambda_mocking_function(orders)