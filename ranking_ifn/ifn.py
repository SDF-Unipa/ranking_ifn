from __future__ import annotations


class IntuitionisticFuzzyNumber:

    def __init__(self, mu: list[float], nu: list[float]) -> None:
        """
        Initialize an intuitionistic fuzzy number with membership and non-membership degrees.

        Args:
        mu (list[float]): Trapezoidal membership parameters [a, b, c, d].
        nu (list[float]): Trapezoidal non-membership parameters [a, b, c, d].
        """

        self.mu_params: list[float] = mu
        self.nu_params: list[float] = nu

        if len(mu) == len(nu) == 4:
            a1, a2, a3, a4 = mu
            b1, b2, b3, b4 = nu
            assert (
                b1 <= a1 and b2 <= a2 <= a3 <= b3 and a4 <= b4
            ), "Invalid trapezoidal parameters for IFN."

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        """
        Compute the alpha-cut of the intuitionistic fuzzy number.

        Args:
        alpha (float): The alpha level for the cut.

        Returns:
        tuple[float, float]: The alpha-cut interval for membership.
        """
        a1, a2, a3, a4 = self.mu_params
        return [a1 + (a2 - a1) * alpha, a4 - (a4 - a3) * alpha]

    def beta_cut(self, beta: float) -> tuple[float, float]:
        """
        Compute the beta-cut of the intuitionistic fuzzy number.

        Args:
        beta (float): The beta level for the cut.

        Returns:
        tuple[float, float]: The beta-cut interval for non-membership.
        """
        b1, b2, b3, b4 = self.nu_params
        return [b2 - (b2 - b1) * beta, b3 + (b4 - b3) * beta]

    def alpha_beta_cut(self, alpha: float, beta: float) -> tuple[float, float]:
        """
        Compute the alpha-beta cut of the intuitionistic fuzzy number.

        Args:
        alpha (float): The alpha level for the membership cut.
        beta (float): The beta level for the non-membership cut.

        Returns:
        tuple[float, float]: The alpha-beta cut intervals for membership and non-membership.
        """
        alpha_cut = self.alpha_cut(alpha)
        beta_cut = self.beta_cut(beta)
        return [max(alpha_cut[0], beta_cut[0]), min(alpha_cut[1], beta_cut[1])]

    def __repr__(self) -> str:
        return f"IFN(mu={self.mu_params}, nu={self.nu_params})"

    def __str__(self) -> str:
        return f"[{tuple(self.mu_params)}, {tuple(self.nu_params)}]"


class IdealIntuitionisticFuzzyNumber(IntuitionisticFuzzyNumber):

    def __init__(self, params: tuple[float, float]) -> None:
        """
        Initialize an ideal intuitionistic fuzzy number.

        Args:
        params (tuple[float, float]): The parameters for the ideal intuitionistic fuzzy number.
        """
        super().__init__([], [])
        self.params: tuple[float, float] = params

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        return [self.params[0], self.params[1]]

    def beta_cut(self, beta: float) -> tuple[float, float]:
        return [self.params[0], self.params[1]]

    def alpha_beta_cut(self, alpha: float, beta: float) -> tuple[float, float]:
        return [self.params[0], self.params[1]]

    def __repr__(self) -> str:
        return f"IIFN(params={self.params})"

    def __str__(self) -> str:
        return f"[{self.params}]"
