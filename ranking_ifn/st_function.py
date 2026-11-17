from __future__ import annotations

import enum

import numpy as np


class STFunctionNames(enum.StrEnum):
    S_FUNCTION_1 = r"\alpha"
    T_FUNCTION_1 = r"1 - \beta"
    S_FUNCTION_2 = r"\alpha \exp(\alpha - 1)"
    T_FUNCTION_2 = r"1 - \beta \exp(\beta - 1)"
    S_FUNCTION_3 = r"\sin\left(\frac{\pi \alpha}{2}\right)"
    T_FUNCTION_3 = r"\cos\left(\frac{\pi \beta}{2}\right)"


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

    @staticmethod
    def all_functions() -> list[tuple[tuple[str, callable], tuple[str, callable]]]:
        """
        Return a list of all S and T functions.

        Returns:
        list[tuple[tuple[str, callable], tuple[str, callable]]]: A list of tuples mapping function names to their corresponding callables.
        """
        return [
            (
                (STFunctionNames.S_FUNCTION_1, STFunction.s_function_1()),
                (STFunctionNames.T_FUNCTION_1, STFunction.t_function_1()),
            ),
            (
                (STFunctionNames.S_FUNCTION_2, STFunction.s_function_2()),
                (STFunctionNames.T_FUNCTION_2, STFunction.t_function_2()),
            ),
            (
                (STFunctionNames.S_FUNCTION_3, STFunction.s_function_3()),
                (STFunctionNames.T_FUNCTION_3, STFunction.t_function_3()),
            ),
        ]
