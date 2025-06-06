# environment/reward_validator_base.py

from .order import Order

class RewardValidatorBase():
    """
    Awards reward for successful or failure order basing on approach defined in
    derivative class.
    """

    def __init__(self, *args) -> None:
        """
        Class constructor. Parameters are specified in derivative classes.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def validate_orders(self, orders: list[Order]) -> float:
        """
        Calculates number of points to be rewarded for list of closed trades.

        Parameters:
            orders (list[Order]): Orders to be validated.

        Returns:
            (float): Calcualted reward.
        """

        raise NotImplementedError("Subclasses must implement this method.")