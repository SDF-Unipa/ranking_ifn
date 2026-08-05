"""Ranking methods used by Popa (2023, doi:10.15837/ijccc.2023.2.5118, Table 8)
for comparison on IFNs, invoked by `test.py` on its examples (Popa's examples 15-21).

Computes `rankings.json` (orderings in `ranks_to_string` format), consumed by
`comparison_table()` in `test.py`. The implemented methods have been verified to
reproduce Popa's Table 8 on all seven examples.

Notation: IFN with membership (a1, a2, a3, a4) and non-membership (b1, b2, b3, b4).
"""

from __future__ import annotations

import json
from pathlib import Path

from ranking_ifn.ifn import IntuitionisticFuzzyNumber
from ranking_ifn.ifn_distance import DATA_DIR, _close, ranks_to_string

# Constants and method metadata
# Rankings are stored in `rankings.json` (created if missing) in the same directory
RANKINGS_PATH: Path = DATA_DIR / "rankings.json"
# Lambda sweep for Popa's method (2023)
LAMBDA_GRID: list[float] = [round(0.1 * k, 1) for k in range(11)]
# Method metadata for `rankings.json` (name, DOI, optional URL)
METHODS: dict[str, dict] = {
    "ye2011": {"name": "Ye's method", "doi": "10.1016/j.eswa.2011.03.059"},
    "rezvani2013": {"name": "Rezvani's method", "doi": None},
    "roseline2013": {
        "name": "Roseline et al.'s method",
        "doi": "10.15373/2249555X/JUNE2013/183",
    },
    "bharati2017": {
        "name": "Bharati's method",
        "doi": None,
        "url": "https://www.ripublication.com/gjpam17/gjpamv13n9_16.pdf",
    },
    "mohan2020": {
        "name": "Mohan et al.'s method",
        "doi": "10.22105/jfea.2020.247301.1003",
    },
    "Popa2023": {"name": "Popa's method", "doi": "10.15837/ijccc.2023.2.5118"},
}


def ye_2011(ifn: IntuitionisticFuzzyNumber) -> float:
    """
    Expected value, Ye (2011) Eq. (20): EV = (sum of all 8 parameters) / 8.

    Args:
        ifn (IntuitionisticFuzzyNumber): The IFN to evaluate.

    Returns:
        float: The expected value of the IFN.
    """
    return (sum(ifn.mu_params) + sum(ifn.nu_params)) / 8


def rezvani_2013(ifn: IntuitionisticFuzzyNumber) -> float:
    """
    Value of the membership function, Rezvani (2013) Theorem 4.3(i):
    V_mu = (a + 2b + 2c + d) / 6.

    Args:
        ifn (IntuitionisticFuzzyNumber): The IFN to evaluate.

    Returns:
        float: The value of the membership function.
    """
    a, b, c, d = ifn.mu_params
    return (a + 2 * b + 2 * c + d) / 6


def roseline_2013(ifn: IntuitionisticFuzzyNumber) -> tuple[float, float]:
    """
    Magnitude, Roseline & Amirtharaj (2013) Eq. (4):
    Mag(A_mu) = (a1 + 5a2 + 5a3 + a4)/12, tie broken by -Mag(A_gamma)
    (Algorithm 4.1: higher -Mag(A_gamma) wins when Mag(A_mu) are equal).

    Args:
        ifn (IntuitionisticFuzzyNumber): The IFN to evaluate.

    Returns:
        tuple[float, float]: The magnitude of the membership function and the
        negative magnitude of the non-membership function.
    """
    a1, a2, a3, a4 = ifn.mu_params
    b1, b2, b3, b4 = ifn.nu_params
    mag_mu = (a1 + 5 * a2 + 5 * a3 + a4) / 12
    mag_gamma = (b1 + 5 * b2 + 5 * b3 + b4) / 12
    return (mag_mu, -mag_gamma)


def bharati_2017(ifn: IntuitionisticFuzzyNumber) -> float:
    """
    Ratio ranking, Bharati (2017): R(A) = ((a4 - a1)/(b4 - b1)) * D^s(A, 0),
    trapezoidal generalization of Sec. 4; D^s is the sign distance from the
    fuzzy origin, which for a trapezoidal IFN equals Ye's expected value.

    Args:
        ifn (IntuitionisticFuzzyNumber): The IFN to evaluate.

    Returns:
        float: The ranking value of the IFN.
    """
    spread_ratio = (ifn.mu_params[3] - ifn.mu_params[0]) / (
        ifn.nu_params[3] - ifn.nu_params[0]
    )
    return spread_ratio * ye_2011(ifn)


def mohan_2020(ifn: IntuitionisticFuzzyNumber) -> float:
    """
    Magnitude, Mohan et al. (2020) Eq. (2):
    Mag(A) = (a1 + b1 + a4 + b4 + 2(a2 + a3 + b2 + b3)) / 12.

    Args:
        ifn (IntuitionisticFuzzyNumber): The IFN to evaluate.

    Returns:
        float: The magnitude of the IFN.
    """
    a1, a2, a3, a4 = ifn.mu_params
    b1, b2, b3, b4 = ifn.nu_params
    return (a1 + b1 + a4 + b4 + 2 * (a2 + a3 + b2 + b3)) / 12


def popa_2023(ifn: IntuitionisticFuzzyNumber, lam: float) -> float:
    """
    Ranking function, Popa (2023) Definition 10 with R(A+) = sum(mu)/4 and
    R(A-) = sum(nu)/4: the optimism weight lam goes to the larger of the two.

    Args:
        ifn (IntuitionisticFuzzyNumber): The IFN to evaluate.
        lam (float): The optimism weight.

    Returns:
        float: The ranking value of the IFN.
    """
    r_plus = sum(ifn.mu_params) / 4
    r_minus = sum(ifn.nu_params) / 4
    if sum(ifn.mu_params) >= sum(ifn.nu_params):
        return lam * r_plus + (1 - lam) * r_minus
    return (1 - lam) * r_plus + lam * r_minus


def ordering(scores: list[float]) -> str:
    """
    Render scores as a `ranks_to_string` ordering, e.g. 'F_{1} < F_{2}'.

    Args:
        scores (list[float]): The scores to rank.

    Returns:
        str: The ordering of the scores as a string.
    """
    indices: list[int] = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )
    # keep index order deterministic inside approximately-equal groups, so that
    # float noise in the sums cannot flip the printed order of tied IFNs
    for i in range(len(indices) - 1):
        if _close(scores[indices[i]], scores[indices[i + 1]]) and (
            indices[i] > indices[i + 1]
        ):
            indices[i], indices[i + 1] = indices[i + 1], indices[i]
    return ranks_to_string(indices, scores)


def rank_example(ifns: list[IntuitionisticFuzzyNumber]) -> dict[str, list[str]]:
    """
    All methods on one example; each value is a list of distinct orderings
    (singleton except for Popa's lambda sweep).

    Args:
        ifns (list[IntuitionisticFuzzyNumber]): The IFNs to rank.

    Returns:
        dict[str, list[str]]: The orderings of the IFNs for each method.
    """
    result = {
        "ye2011": [ordering([ye_2011(f) for f in ifns])],
        "rezvani2013": [ordering([rezvani_2013(f) for f in ifns])],
        "roseline2013": [ordering([roseline_2013(f) for f in ifns])],
        "bharati2017": [ordering([bharati_2017(f) for f in ifns])],
        "mohan2020": [ordering([mohan_2020(f) for f in ifns])],
    }
    popa: list[str] = []
    for lam in LAMBDA_GRID:
        s = ordering([popa_2023(f, lam) for f in ifns])
        if s not in popa:
            popa.append(s)
    result["Popa2023"] = popa
    return result


def write_rankings(example: int, ranked: dict[str, list[str]]) -> None:
    """
    Update one example entry in rankings.json (created if missing).

    Args:
        example (int): The example number (1-7).
        ranked (dict[str, list[str]]): The orderings of the IFNs for each method.
    """
    if RANKINGS_PATH.exists():
        with open(RANKINGS_PATH, encoding="utf-8") as f:
            rankings = json.load(f)
    else:
        rankings = {"examples": {}}
    rankings["source"] = (
        "Computed by ranking_ifn/comparison_methods.py"
    )
    rankings["methods"] = METHODS
    rankings["examples"][str(example)] = {
        **ranked,
    }
    rankings["examples"] = dict(
        sorted(rankings["examples"].items(), key=lambda kv: int(kv[0]))
    )
    with open(RANKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(rankings, f, ensure_ascii=False, indent=4, sort_keys=True)


def print_comparison(ifns: list[IntuitionisticFuzzyNumber], example: int) -> None:
    """
    Print the orderings of Popa's Table 8 comparison methods for one example
    and update its entry in rankings.json.

    Args:
        ifns (list[IntuitionisticFuzzyNumber]): The IFNs to rank.
        example (int): The example number (1-7).
    """
    ranked = rank_example(ifns)
    for method, orderings in ranked.items():
        print(f"  {method:>13}: {', '.join(orderings)}")
    write_rankings(example, ranked)
