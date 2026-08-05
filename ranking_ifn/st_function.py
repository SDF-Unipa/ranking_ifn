from __future__ import annotations

import numpy as np


class STFunction:

    @staticmethod
    def s_function_1() -> callable:
        """
        Return the first S-function.

        Returns:
        callable: The first S-function.
        """
        return lambda alpha: alpha

    @staticmethod
    def t_function_1() -> callable:
        """
        Return the first T-function.

        Returns:
        callable: The first T-function.
        """
        return lambda beta: 1 - beta

    @staticmethod
    def s_function_2() -> callable:
        """
        Return the second S-function.

        Returns:
        callable: The second S-function.
        """
        return lambda alpha: alpha * np.exp(alpha - 1)

    @staticmethod
    def t_function_2() -> callable:
        """
        Return the second T-function.

        Returns:
        callable: The second T-function.
        """
        return lambda beta: 1 - beta * np.exp(beta - 1)

    @staticmethod
    def s_function_3() -> callable:
        """
        Return the third S-function.

        Returns:
        callable: The third S-function.
        """
        return lambda alpha: np.sin(np.pi * alpha / 2)

    @staticmethod
    def t_function_3() -> callable:
        """
        Return the third T-function.

        Returns:
        callable: The third T-function.
        """
        return lambda beta: np.cos(np.pi * beta / 2)
