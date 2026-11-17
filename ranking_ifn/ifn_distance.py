from __future__ import annotations

import logging
import math
import typing
from pathlib import Path

import numpy as np
from scipy.special import expit

from ranking_ifn.ifn import IdealIntuitionisticFuzzyNumber, IntuitionisticFuzzyNumber
from ranking_ifn.st_function import STFunctionNames

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intuitionistic.log", mode="w")],
)
logger = logging.getLogger(__name__)


# Create a data directory for storing ranking index results
DATA_DIR: Path = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def safe_ratio(num: np.ndarray, other: np.ndarray) -> np.ndarray:
    """
    Calculate the ratio of two arrays, handling division by zero by returning 0.5 when the denominator is zero.

    Args:
    num (np.ndarray): The numerator array.
    other (np.ndarray): The denominator array.

    Returns:
    np.ndarray: The resulting array after division, with 0.5 where the denominator is zero.
    """
    return np.where(other == 0.0, 0.5, num / np.where(other == 0.0, 1.0, other))


def update_data_file(file_path: typing.Optional[Path], data: dict) -> None:
    """
    Update the JSON file with new data. If the file does not exist, it will be created.

    Args:
    file_path (Path | None): The path to the JSON file. If None, the file will not be updated.
    data (dict): The data to be added to the JSON file.
    """
    import json

    if file_path is None:
        # If no file path is provided, do not attempt to update the file.
        return

    if not file_path.exists():
        existing_data: list = []
    else:
        with file_path.open("r") as f:
            try:
                existing_data: list = json.load(f)
            except json.JSONDecodeError:
                existing_data: list = []

    with file_path.open("w") as f:
        json.dump(existing_data + [data], f, indent=4, sort_keys=True)


def _close(x: float | tuple[float, ...], y: float | tuple[float, ...]) -> bool:
    """
    Check if two values (or tuples of values) are approximately equal within a specified tolerance.

    Args:
    x (float | tuple[float, ...]): The first value or tuple of values.
    y (float | tuple[float, ...]): The second value or tuple of values.

    Returns:
    bool: True if the values are approximately equal, False otherwise.
    """
    if isinstance(x, tuple):
        return all(_close(a, b) for a, b in zip(x, y))
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)


def ranks_to_string(indices: np.ndarray, ranks: np.ndarray) -> str:
    """
    Convert a list of ranks (indices) to a string representation.

    Args:
    indices (np.ndarray): The array of indices representing the ranks.
    ranks (np.ndarray): The array of ranking values.

    Returns:
    str: A string representation of the ranks.
    """

    ranks_str: list[str] = [f"F_{{{i+1}}}" for i in indices]
    if len(ranks_str) == 1:
        # If there is only one rank, return it directly
        return ranks_str[0]
    if len(ranks_str) == 2:
        # If there are two ranks, compare them and return the appropriate string representation
        if _close(ranks[indices[0]], ranks[indices[1]]):
            # If the two ranks are approximately equal, return a string indicating that they are approximately equal
            return f"{ranks_str[0]} \\approx {ranks_str[1]}"
        elif indices[0] < indices[1]:
            # If the first index is less than the second index, return a string indicating that the first rank is greater than the second rank
            return f"{ranks_str[0]} > {ranks_str[1]}"
        else:
            # If the first index is greater than the second index, return a string indicating that the first rank is less than the second rank
            return f"{ranks_str[1]} < {ranks_str[0]}"

    rank_comparison: list[str] = [
        "\\approx" if _close(ranks[indices[i]], ranks[indices[i + 1]]) else ">"
        for i in range(len(indices) - 1)
    ]
    result: str = (
        " ".join(
            f"{ranks_str[i]} {rank_comparison[i]}" for i in range(len(rank_comparison))
        )
        + " "
        + ranks_str[-1]
    )
    return result


class DistanceCalculator:

    @staticmethod
    def interval_distance(
        interval1: tuple[float, float],
        interval2: tuple[float, float],
    ) -> float:
        """
        Calculate the distance between two intervals.

        Args:
        interval1 (tuple[float, float]): The first interval.
        interval2 (tuple[float, float]): The second interval.

        Returns:
        float: The distance between the two intervals.
        """
        al, ar = interval1
        bl, br = interval2

        if al == bl and ar == br:
            result: float = 0.0
        elif (al, bl) == (ar, br):
            result: float = np.abs(al - bl)
        elif al != ar and bl != br:
            result: float = (
                np.abs(ar - bl) ** 3
                + np.abs(al - br) ** 3
                - np.abs(ar - br) ** 3
                - np.abs(al - bl) ** 3
            ) / (6 * (ar - al) * (br - bl))
        elif al == ar and bl != br:
            result: float = (
                np.abs(al - bl) * (al - bl) - np.abs(al - br) * (al - br)
            ) / (2 * (br - bl))
        elif al != ar and bl == br:
            result: float = (
                np.abs(ar - bl) * (ar - bl) - np.abs(al - bl) * (al - bl)
            ) / (2 * (ar - al))
        return result

    @staticmethod
    def fuzzy_membership_distance(
        s_function: callable,
        fuzzy1: IntuitionisticFuzzyNumber,
        fuzzy2: IntuitionisticFuzzyNumber,
        n: int = 100,
    ) -> float:
        """
        Calculate the fuzzy membership distance between two intuitionistic fuzzy numbers.

        Args:
        s_function (callable): The S-function for membership.
        fuzzy1 (IntuitionisticFuzzyNumber): The first intuitionistic fuzzy number.
        fuzzy2 (IntuitionisticFuzzyNumber): The second intuitionistic fuzzy number.
        n (int): The number of points for numerical integration (default is 100).

        Returns:
        float: The membership distance between the two intuitionistic fuzzy numbers.
        """
        alpha_levels: np.ndarray = np.linspace(0, 1.0, n)
        s_values: np.ndarray = np.array([s_function(alpha) for alpha in alpha_levels])
        distance_intervals: np.ndarray = np.array(
            list(
                map(
                    lambda alpha: DistanceCalculator.interval_distance(
                        fuzzy1.alpha_cut(alpha), fuzzy2.alpha_cut(alpha)
                    ),
                    alpha_levels,
                )
            )
        )
        logger.debug(
            f"Interval distance between {fuzzy1} and {fuzzy2}: {distance_intervals}"
        )
        result: float = np.trapezoid(
            s_values * distance_intervals, alpha_levels
        ) / np.trapezoid(s_values, alpha_levels)
        return result

    @staticmethod
    def fuzzy_non_membership_distance(
        t_function: callable,
        fuzzy1: IntuitionisticFuzzyNumber,
        fuzzy2: IntuitionisticFuzzyNumber,
        m: int = 100,
    ) -> float:
        """
        Calculate the distance between the non-membership functions of two intuitionistic fuzzy numbers.

        Args:
        t_function (callable): The T-function for non-membership.
        fuzzy1 (IntuitionisticFuzzyNumber): The first intuitionistic fuzzy number.
        fuzzy2 (IntuitionisticFuzzyNumber): The second intuitionistic fuzzy number.
        m (int): The number of points for numerical integration (default is 100).

        Returns:
        float: The distance between the non-membership functions of the two intuitionistic fuzzy numbers.
        """
        beta_levels: np.ndarray = np.linspace(0, 1.0, m)
        t_values: np.ndarray = np.array([t_function(beta) for beta in beta_levels])
        distance_intervals: np.ndarray = np.array(
            list(
                map(
                    lambda beta: DistanceCalculator.interval_distance(
                        fuzzy1.beta_cut(beta), fuzzy2.beta_cut(beta)
                    ),
                    beta_levels,
                )
            )
        )
        logger.debug(
            f"Interval distance between {fuzzy1} and {fuzzy2}: {distance_intervals}"
        )
        result: float = np.trapezoid(
            t_values * distance_intervals, beta_levels
        ) / np.trapezoid(t_values, beta_levels)
        return result

    @staticmethod
    def fuzzy_membership_non_membership_distance(
        s_function: callable,
        t_function: callable,
        fuzzy1: IntuitionisticFuzzyNumber,
        fuzzy2: IntuitionisticFuzzyNumber,
        n: int = 100,
        m: int = 100,
    ) -> float:
        """
        Calculate the distance between two intuitionistic fuzzy numbers considering both membership and non-membership functions.

        Args:
        s_function (callable): The S-function for membership.
        t_function (callable): The T-function for non-membership.
        fuzzy1 (IntuitionisticFuzzyNumber): The first intuitionistic fuzzy number.
        fuzzy2 (IntuitionisticFuzzyNumber): The second intuitionistic fuzzy number.
        n (int): The number of points for numerical integration (default is 100).
        m (int): The number of points for numerical integration (default is 100).

        Returns:
        float: The distance between the two intuitionistic fuzzy numbers considering both membership and non-membership functions.
        """
        alpha_levels: np.ndarray = np.linspace(0, 1.0, n)
        beta_levels: np.ndarray = np.linspace(0, 1.0, m)

        # s(alpha) for all alpha in [0,1]
        s_values: np.ndarray = np.array(
            list(map(lambda alpha: s_function(alpha), alpha_levels))
        )
        # t(beta) for all beta in [0,1]
        t_values: np.ndarray = np.array(
            list(map(lambda beta: t_function(beta), beta_levels))
        )

        # alphas[i,j] = alpha_j
        # betas[i,j] = beta_i
        alphas, betas = np.meshgrid(alpha_levels, beta_levels)
        # mask[i,j] = True if alphas[i,j] + betas[i,j] <= 1.0, else False
        mask: np.ndarray = alphas + betas <= 1.0

        # Gamma(alphas, betas) = s(alpha_j) * t(beta_i) for all i,j
        gamma: np.ndarray = np.outer(t_values, s_values)

        # distance between the alpha-beta cuts of the two intuitionistic fuzzy numbers
        # d_I(fuzzy1, fuzzy2) = d_I(fuzzy1.alpha_beta_cut(alpha_j, beta_i), fuzzy2.alpha_beta_cut(alpha_j, beta_i))
        distance_intervals_alpha_beta: np.ndarray = np.array(
            list(
                map(
                    lambda beta: list(
                        map(
                            lambda alpha: DistanceCalculator.interval_distance(
                                fuzzy1.alpha_beta_cut(alpha, beta),
                                fuzzy2.alpha_beta_cut(alpha, beta),
                            ),
                            alpha_levels,
                        )
                    ),
                    beta_levels,
                )
            )
        )

        logger.debug(
            f"Interval distance between {fuzzy1} and {fuzzy2}: {distance_intervals_alpha_beta}"
        )

        # double integration in [0, 1]^2 with the constraint alpha + beta <= 1.0
        # d_{\mu, \nu} = \frac{\int_0^1 \int_0^{1} \gamma(\alpha, \beta) d_I(fuzzy1, fuzzy2) d\alpha d\beta}{\int_0^1 \int_0^{1} \gamma(\alpha, \beta) d\alpha d\beta}

        # put 0.0 in the positions where alpha + beta > 1.0
        gamma_triangular: np.ndarray = np.where(mask, gamma, 0.0)
        result: float = np.trapezoid(
            np.trapezoid(
                gamma_triangular * distance_intervals_alpha_beta, beta_levels, axis=0
            ),
            alpha_levels,
            axis=0,
        ) / np.trapezoid(
            np.trapezoid(gamma_triangular, beta_levels, axis=0), alpha_levels, axis=0
        )
        return result

    @staticmethod
    def compute_distances(
        s_function: callable,
        t_function: callable,
        fuzzy1: IntuitionisticFuzzyNumber,
        fuzzy2: tuple[
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
        ],
        n: int = 100,
        m: int = 100,
    ) -> tuple[float, float, float]:
        """
        Compute the membership distance, non-membership distance, and membership-non-membership distance between two intuitionistic fuzzy numbers.

        Args:
        s_function (callable): The S-function for membership.
        t_function (callable): The T-function for non-membership.
        fuzzy1 (IntuitionisticFuzzyNumber): The first intuitionistic fuzzy number.
        fuzzy2 (tuple[IdealIntuitionisticFuzzyNumber, IdealIntuitionisticFuzzyNumber, IdealIntuitionisticFuzzyNumber]): The second intuitionistic fuzzy number.
        n (int): The number of points for numerical integration (default is 100).
        m (int): The number of points for numerical integration (default is 100).

        Returns:
        tuple[float, float, float]: A tuple containing the membership distance, non-membership distance, and membership-non-membership distance.
        """

        logger.debug(f"Fuzzy1: {fuzzy1}, Fuzzy2: {fuzzy2}")
        logger.debug(f"Parameters: n={n}, m={m}")
        logger.debug("Calculating membership distance...")
        membership_distance: float = DistanceCalculator.fuzzy_membership_distance(
            s_function, fuzzy1, fuzzy2[0], n=n
        )
        logger.debug(f"Membership distance: {membership_distance}")
        logger.debug("Calculating non-membership distance...")
        non_membership_distance: float = (
            DistanceCalculator.fuzzy_non_membership_distance(
                t_function, fuzzy1, fuzzy2[1], m=m
            )
        )
        logger.debug(f"Non-membership distance: {non_membership_distance}")
        logger.debug("Calculating membership-non-membership distance...")
        membership_non_membership_distance: float = (
            DistanceCalculator.fuzzy_membership_non_membership_distance(
                s_function, t_function, fuzzy1, fuzzy2[2], n=n, m=m
            )
        )
        return (
            membership_distance,
            non_membership_distance,
            membership_non_membership_distance,
        )

    @staticmethod
    def f_ast(fuzzy: list[IntuitionisticFuzzyNumber]) -> list[
        tuple[
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
        ]
    ]:
        """
        Calculate the F* values for a list of intuitionistic fuzzy numbers.

        Args:
        fuzzy (list[IntuitionisticFuzzyNumber]): A list of intuitionistic fuzzy numbers.

        Returns:
        list[tuple[IdealIntuitionisticFuzzyNumber, IdealIntuitionisticFuzzyNumber, IdealIntuitionisticFuzzyNumber]]: A list of F* values for each intuitionistic fuzzy number.
        """

        # L_i^a(\alpha) =  a_1^i + alpha * (a_2^i - a_1^i)
        # R_i^a(\alpha) = a_4^i - alpha * (a_4^i - a_3^i)
        # L_i^b(\beta) = b_2^i - beta * (b_2^i - b_1^i)
        # R_i^b(\beta) = b_3^i + beta * (b_4^i - b_3^i)

        # L-R functions are affine functions, so the suprema and the infima are at the endpoints of the intervals.

        # sup_i L_i^a(alpha) = sup_i L_i^a(0) = \sup_i a_1^i
        # inf_i L_i^a(alpha) = inf_i L_i^a(0) = \inf_i a_1^i
        # sup_i R_i^a(alpha) = sup_i R_i^a(0) = \sup_i a_4^i
        # inf_i R_i^a(alpha) = inf_i R_i^a(0) = \inf_i a_4^i
        # sup_i L_i^b(beta) = sup_i L_i^b(1) = \sup_i b_1^i
        # inf_i L_i^b(beta) = inf_i L_i^b(1) = \inf_i b_1^i
        # sup_i R_i^b(beta) = sup_i R_i^b(1) = \sup_i b_4^i
        # inf_i R_i^b(beta) = inf_i R_i^b(1) = \inf_i b_4^i

        # lower_mu_values = [a_1^i for i in range(len(fuzzy))]
        lower_mu_values: np.ndarray = np.array(
            [fuzzy[i].mu_params[0] for i in range(len(fuzzy))]
        )
        # upper_mu_values = [a_4^i for i in range(len(fuzzy))]
        upper_mu_values: np.ndarray = np.array(
            [fuzzy[i].mu_params[3] for i in range(len(fuzzy))]
        )
        # lower_nu_values = [b_1^i for i in range(len(fuzzy))]
        lower_nu_values: np.ndarray = np.array(
            [fuzzy[i].nu_params[0] for i in range(len(fuzzy))]
        )
        # upper_nu_values = [b_4^i for i in range(len(fuzzy))]
        upper_nu_values: np.ndarray = np.array(
            [fuzzy[i].nu_params[3] for i in range(len(fuzzy))]
        )

        # F^\ast_\mu = \bigcup_{\alpha \in [0, 1]} [sup_i L_i^a(alpha), sup_i R_i^a(alpha)] =
        #             = [sup_i L_i^a(0), sup_i R_i^a(0)] = [sup_i a_1^i, sup_i a_4^i]
        f_ast_mu: tuple[float, float] = (
            np.max(lower_mu_values),
            np.max(upper_mu_values),
        )

        # F^\ast_\nu = \bigcup_{\beta \in [0, 1]} [sup_i L_i^b(beta), sup_i R_i^b(beta)] =
        #             = [sup_i L_i^b(1), sup_i R_i^b(1)] = [sup_i b_1^i, sup_i b_4^i]
        f_ast_nu: tuple[float, float] = (
            np.max(lower_nu_values),
            np.max(upper_nu_values),
        )

        # F^\mu_\ast = \bigcup_{\alpha \in [0, 1]} [inf_i L_i^a(alpha), inf_i R_i^a(alpha)] =
        #             = [inf_i L_i^a(0), inf_i R_i^a(0)] = [inf_i a_1^i, inf_i a_4^i]
        f_mu_ast: tuple[float, float] = (
            np.min(lower_mu_values),
            np.min(upper_mu_values),
        )

        # F^\ast_\nu = \bigcup_{\beta \in [0, 1]} [inf_i L_i^b(beta), inf_i R_i^b(beta)] =
        #             = [inf_i L_i^b(1), inf_i R_i^b(1)] = [inf_i b_1^i, inf_i b_4^i]
        f_nu_ast: tuple[float, float] = (
            np.min(lower_nu_values),
            np.min(upper_nu_values),
        )

        # F^\ast_{\mu\nu} = [max(sup_i L_i(alpha), inf_i L_i(beta)), min(sup_i R_i(alpha), inf_i R_i(beta))]
        #                  = [max(sup_i a_1^i, inf_i b_1^i), min(sup_i a_4^i, inf_i b_4^i)]
        f_ast_mu_nu: tuple[float, float] = (
            np.maximum(f_ast_mu[0], f_ast_nu[0]),
            np.minimum(f_ast_mu[1], f_ast_nu[1]),
        )
        # F^{\mu\nu}_\ast = [max(inf_i L_i(alpha), sup_i L_i(beta)), min(inf_i R_i(alpha), sup_i R_i(beta))]
        #                  = [max(inf_i a_1^i, sup_i b_1^i), min(inf_i a_4^i, sup_i b_4^i)]
        f_mu_nu_ast: tuple[float, float] = (
            np.maximum(f_mu_ast[0], f_nu_ast[0]),
            np.minimum(f_mu_ast[1], f_nu_ast[1]),
        )

        result: list[
            tuple[
                IdealIntuitionisticFuzzyNumber,
                IdealIntuitionisticFuzzyNumber,
                IdealIntuitionisticFuzzyNumber,
            ]
        ] = [
            (
                IdealIntuitionisticFuzzyNumber(f_ast_mu),
                IdealIntuitionisticFuzzyNumber(f_ast_nu),
                IdealIntuitionisticFuzzyNumber(f_ast_mu_nu),
            ),
            (
                IdealIntuitionisticFuzzyNumber(f_mu_ast),
                IdealIntuitionisticFuzzyNumber(f_nu_ast),
                IdealIntuitionisticFuzzyNumber(f_mu_nu_ast),
            ),
        ]
        logger.debug(
            f"$F^\ast_\mu$ = {f_ast_mu}, $F^\ast_\\nu$ = {f_ast_nu}, $F^\ast_{{\mu\\nu}}$ = {f_ast_mu_nu}"
        )
        logger.debug(
            f"$F^\mu_\ast$ = {f_mu_ast}, $F^\\nu_\ast$ = {f_nu_ast}, $F^{{\mu\\nu}}_\ast$ = {f_mu_nu_ast}"
        )
        return result

    @staticmethod
    def sigma_function(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate the sigma function for two numpy arrays.
        The sigma function is defined as the softmax of the element-wise differences between the two arrays.

        In particular, this function computes:
        r[i] = exp(a[i]) / (exp(a[i]) + exp(b[i])) = 1 / (1 + exp(b[i] - a[i])) = \\sigma(a[i] - b[i]),
        where \\sigma(x) = 1 / (1 + exp(-x)) is the sigmoid function.

        Args:
        a (np.ndarray): The first array.
        b (np.ndarray): The second array.

        Returns:
        np.ndarray: The result of the sigma function.
        """
        # Compute the element-wise sigma function using the sigmoid of the differences between a and b.
        return expit(a - b)

    @staticmethod
    def ranking_index(
        s_function: callable,
        t_function: callable,
        fuzzy: list[IntuitionisticFuzzyNumber],
        n: int = 100,
        m: int = 100,
        file_path: Path = DATA_DIR / "ranking_index.json",
        s_function_name: str = STFunctionNames.S_FUNCTION_1,
        t_function_name: str = STFunctionNames.T_FUNCTION_1,
    ) -> np.ndarray:
        """
        Calculate the ranking index for a list of intuitionistic fuzzy numbers.

        Args:
        fuzzy (list[IntuitionisticFuzzyNumber]): A list of intuitionistic fuzzy numbers.
        n (int): The number of points for numerical integration (default is 100).
        m (int): The number of points for numerical integration (default is 100).
        file_path (Path): The path to the file where the ranking index will be saved (default is "ranking_index.json").
        s_function_name (str): The name of the S-function (default is STFunctionNames.S_FUNCTION_1).
        t_function_name (str): The name of the T-function (default is STFunctionNames.T_FUNCTION_1).

        Returns:
        np.ndarray: An array of ranking index values for each intuitionistic fuzzy number.
        """
        logger.debug("Calculating ranking index for intuitionistic fuzzy numbers.")
        logger.debug(f"Parameters: n={n}, m={m}")
        logger.debug(f"Fuzzy numbers: {fuzzy}")
        logger.debug("Calculating F* values...")

        f_ast_values: list[
            tuple[
                IdealIntuitionisticFuzzyNumber,
                IdealIntuitionisticFuzzyNumber,
                IdealIntuitionisticFuzzyNumber,
            ]
        ] = DistanceCalculator.f_ast(fuzzy)
        f_ast_all: tuple[
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
        ] = (f_ast_values[0][0], f_ast_values[0][1], f_ast_values[0][2])
        f_all_ast: tuple[
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
            IdealIntuitionisticFuzzyNumber,
        ] = (f_ast_values[1][0], f_ast_values[1][1], f_ast_values[1][2])

        logger.debug(
            "Calculating fuzzy distances for F^\ast_\mu, F^\ast_\\nu, and F^\ast_{\mu\\nu}..."
        )
        (
            membership_distances_ast_mu,
            non_membership_distances_ast_nu,
            membership_non_membership_distances_ast_mu_nu,
        ) = zip(
            *[
                DistanceCalculator.compute_distances(
                    s_function, t_function, fuzz, f_ast_all, n=n, m=m
                )
                for fuzz in fuzzy
            ]
        )
        logger.debug(
            "Calculating fuzzy distances for F^\mu_\ast, F_\ast^\\nu, and F^{\mu\\nu}_\ast..."
        )
        (
            membership_distances_mu_ast,
            non_membership_distances_nu_ast,
            membership_non_membership_distances_mu_nu_ast,
        ) = zip(
            *[
                DistanceCalculator.compute_distances(
                    s_function, t_function, fuzz, f_all_ast, n=n, m=m
                )
                for fuzz in fuzzy
            ]
        )
        # Convert the distances to numpy arrays for easier manipulation
        membership_distances_ast_mu = np.array(membership_distances_ast_mu)
        non_membership_distances_ast_nu = np.array(non_membership_distances_ast_nu)
        membership_non_membership_distances_ast_mu_nu = np.array(
            membership_non_membership_distances_ast_mu_nu
        )
        membership_distances_mu_ast = np.array(membership_distances_mu_ast)
        non_membership_distances_nu_ast = np.array(non_membership_distances_nu_ast)
        membership_non_membership_distances_mu_nu_ast = np.array(
            membership_non_membership_distances_mu_nu_ast
        )

        logger.debug(
            f"Membership distances (F^\ast_\\mu): {membership_distances_ast_mu}"
        )
        logger.debug(
            f"Non-membership distances (F^\ast_\\nu): {non_membership_distances_ast_nu}"
        )
        logger.debug(
            f"Membership-non-membership distances (F^\ast_{{\\mu\\nu}}): {membership_non_membership_distances_ast_mu_nu}"
        )
        logger.debug(
            f"Membership distances (F^\\mu_\\ast): {membership_distances_mu_ast}"
        )
        logger.debug(
            f"Non-membership distances (F_\\ast^\\nu): {non_membership_distances_nu_ast}"
        )
        logger.debug(
            f"Membership-non-membership distances (F^{{\\mu\\nu}}_\\ast): {membership_non_membership_distances_mu_nu_ast}"
        )

        delta_mu: np.ndarray = membership_distances_ast_mu - membership_distances_mu_ast
        delta_nu: np.ndarray = (
            non_membership_distances_ast_nu - non_membership_distances_nu_ast
        )

        # Compute the sigma factors using the sigmoid function of the differences between delta_mu and delta_nu.
        # sigma_i = sigmoid(delta_mu_i - delta_nu_i) for each i = 1 / (1 + exp(-(delta_mu_i - delta_nu_i)))
        sigma_factors: np.ndarray = np.array(
            DistanceCalculator.sigma_function(delta_mu, delta_nu)
        )

        # R_i = 1/2 * (
        #           d_{\mu \nu}(F_i, F^{\mu \nu}_\ast) / (d_{\mu \nu}(F_i, F^{\mu \nu}_\ast) + d_{\mu \nu}(F_i, F^\ast_{\mu \nu})) +
        #           \sigma_i * (d_\mu(F_i, F^\ast_\mu) / (d_\mu(F_i, F^\ast_\mu) + d_\mu(F_i, F^\mu_\ast))) +
        #           (1 - \sigma_i) * (d_\nu(F_i, F^\ast_\nu) / (d_\nu(F_i, F^\ast_\nu) + d_\nu(F_i, F^\nu_\ast)))
        #       )

        # first_term = d_{\mu \nu}(F_i, F^{\mu \nu}_\ast) / (d_{\mu \nu}(F_i, F^{\mu \nu}_\ast) + d_{\mu \nu}(F_i, F^\ast_{\mu \nu}))
        first_term: np.ndarray = safe_ratio(
            membership_non_membership_distances_mu_nu_ast,
            membership_non_membership_distances_ast_mu_nu
            + membership_non_membership_distances_mu_nu_ast,
        )
        # second_term = (d_\mu(F_i, F^\ast_\mu) / (d_\mu(F_i, F^\ast_\mu) + d_\mu(F_i, F^\mu_\ast)))
        second_term: np.ndarray = safe_ratio(
            membership_distances_mu_ast,
            membership_distances_ast_mu + membership_distances_mu_ast,
        )
        # third_term = (d_\nu(F_i, F^\ast_\nu) / (d_\nu(F_i, F^\ast_\nu) + d_\nu(F_i, F^\nu_\ast)))
        third_term: np.ndarray = safe_ratio(
            non_membership_distances_nu_ast,
            non_membership_distances_ast_nu + non_membership_distances_nu_ast,
        )
        # Calculate the ranking index
        ranking_index: np.ndarray = (
            first_term + sigma_factors * second_term + (1 - sigma_factors) * third_term
        ) / 2.0

        logger.debug(f"Ranking index: {ranking_index}")

        # Save the results to a JSON file for further analysis
        update_data_file(
            file_path,
            {
                r"$F_i$": [
                    (tuple(fuzzy[i].mu_params), tuple(fuzzy[i].nu_params))
                    for i in range(len(fuzzy))
                ],
                r"$\delta_mu$": delta_mu.tolist(),
                r"$\delta_nu$": delta_nu.tolist(),
                r"$d_\mu(F_i, F^\ast_mu)$": membership_distances_ast_mu.tolist(),
                r"$d_\nu(F_i, F^\ast_\nu)$": non_membership_distances_ast_nu.tolist(),
                r"$d_{\mu\nu}(F_i, F^\ast_{\mu\nu})$": membership_non_membership_distances_ast_mu_nu.tolist(),
                r"$d_\mu(F_i, F_\ast^\mu)$": membership_distances_mu_ast.tolist(),
                r"$d_\nu(F_i, F_\ast^\nu)$": non_membership_distances_nu_ast.tolist(),
                r"$d_{\mu\nu}(F_i, F_\ast^{\mu\nu})$": membership_non_membership_distances_mu_nu_ast.tolist(),
                r"$\sigma$": sigma_factors.tolist(),
                r"$P_1$": first_term.tolist(),
                r"$P_2$": second_term.tolist(),
                r"$P_3$": third_term.tolist(),
                "$R$": ranking_index.tolist(),
                r"$s$-function": s_function_name,
                r"$t$-function": t_function_name,
            },
        )
        return ranking_index

    @staticmethod
    def ranking(
        s_function: callable,
        t_function: callable,
        fuzzy: list[IntuitionisticFuzzyNumber],
        n: int = 100,
        m: int = 100,
        file_path: Path = DATA_DIR / "ranking_index.json",
        s_function_name: str = STFunctionNames.S_FUNCTION_1,
        t_function_name: str = STFunctionNames.T_FUNCTION_1,
    ) -> tuple[np.ndarray, str]:
        """
        Calculate the ranking index for a list of intuitionistic fuzzy numbers.

        Args:
        s_function (callable): The S-function for membership.
        t_function (callable): The T-function for non-membership.
        fuzzy (list[IntuitionisticFuzzyNumber]): A list of intuitionistic fuzzy numbers.
        n (int): The number of points for numerical integration (default is 100).
        m (int): The number of points for numerical integration (default is 100).
        file_path (Path): The path to the file where the ranking index will be saved (default is "ranking_index.json").
        s_function_name (str): The name of the S-function used (default is STFunctionNames.S_FUNCTION_1).
        t_function_name (str): The name of the T-function used (default is STFunctionNames.T_FUNCTION_1).

        Returns:
        tuple[np.ndarray, str]: A tuple containing the array of ranking index values and a string representation of their ranks.
        """
        logger.debug("Calculating ranking for intuitionistic fuzzy numbers.")
        ranks: np.ndarray = DistanceCalculator.ranking_index(
            s_function,
            t_function,
            fuzzy,
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        indices: np.ndarray = np.argsort(ranks)[::-1]
        return (
            ranks,
            ranks_to_string(indices, ranks),
        )
