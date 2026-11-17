# ranking_ifn

Companion repository for the paper **An Interval-Based Distance for Ranking
Intuitionistic Fuzzy Numbers**. It implements the proposed
distance-based ranking of trapezoidal intuitionistic fuzzy numbers (TrIFN) and
reproduces the comparison against several ranking methods from the literature.
A citation entry and DOI will be added upon publication.

## Method

Distances between intervals use the closed-form interval distance $d_I$
(double integral of $\lvert A(x) - B(y)\rvert$, reduced to a four-vertex
formula; $O(1)$ per evaluation, valid also for degenerate and reversed/Kaucher
intervals, with $d_I(X, X) = 0$). From it, three distances between TrIFNs are
built:

- $d_\mu$ — over $\alpha$-cuts, weighted by a reducing function $s(\alpha)$;
- $d_\nu$ — over $\beta$-cuts, weighted by a growing function $t(\beta)$;
- $d_{\mu\nu}$ — over $(\alpha,\beta)$-cuts, weighted by
  $\Gamma(\alpha,\beta) = s(\alpha)\,t(\beta)$ on the simplex
  $T = \{(\alpha,\beta) : \alpha + \beta \le 1\}$.

The set being ranked induces an ideal and an anti-ideal crisp TrIFN

```math
F^\ast = \big([A_1, A_1, A_4, A_4],\, [B_1, B_1, B_4, B_4]\big),
\qquad
F_\ast = \big([\underline{A}_1, \underline{A}_1, \underline{A}_4, \underline{A}_4],\,
              [\underline{B}_1, \underline{B}_1, \underline{B}_4, \underline{B}_4]\big),
```

where $A_1, A_4, B_1, B_4$ are the suprema over the set of the support
endpoints $a_1^i, a_4^i, b_1^i, b_4^i$, and the underlined counterparts the
infima ($O(n)$). Their cut sets are constant —
$(F^\ast)^\alpha = (F^\ast)^{(\alpha,\beta)} = [A_1, A_4]$ and
$(F^\ast)_\beta = [B_1, B_4]$ — so both are genuine TrIFNs and every distance
involved is a distance between IFNs.

For each number $F_i$ the calculator produces the differences

```math
\Delta_\mu(F_i) = d_\mu(F_i, F^\ast) - d_\mu(F_i, F_\ast),
\qquad
\Delta_\nu(F_i) = d_\nu(F_i, F^\ast) - d_\nu(F_i, F_\ast),
```

the weight function $\sigma(F_i) \in [0, 1]$, continuous in the relative
magnitude of the two differences,

```math
\sigma(F_i) =
\begin{cases}
    \left\vert\Delta_\mu(F_i)/\Delta_\nu(F_i)\right\vert     & \text{if } \vert\Delta_\mu(F_i)\vert < \vert\Delta_\nu(F_i)\vert;\\
    1 - \left\vert\Delta_\nu(F_i)/\Delta_\mu(F_i)\right\vert & \text{if } \vert\Delta_\mu(F_i)\vert > \vert\Delta_\nu(F_i)\vert;\\
    \frac{1}{2} & \text{if } \vert\Delta_\mu(F_i)\vert = \vert\Delta_\nu(F_i)\vert;
\end{cases}
```

and the ranking index $\mathcal{R}_i \in [0, 1]$ (TOPSIS-type, higher ranks
higher):

```math
\mathcal{R}_i = \frac{1}{2}\left[
\frac{d_{\mu\nu}(F_i, F_\ast)}{d_{\mu\nu}(F_i, F^\ast) + d_{\mu\nu}(F_i, F_\ast)}
+ \sigma(F_i)\,\frac{d_\mu(F_i, F_\ast)}{d_\mu(F_i, F^\ast) + d_\mu(F_i, F_\ast)}
+ \left(1-\sigma(F_i)\right)\frac{d_\nu(F_i, F_\ast)}{d_\nu(F_i, F^\ast) + d_\nu(F_i, F_\ast)}
\right]
```

The three ratios share the same pair of ideal elements but differ in the cut
sets of $F_i$ they involve ($\alpha$-, $\beta$-, and $(\alpha,\beta)$-cuts,
respectively). A ratio whose denominator vanishes ($F_i$ coinciding with both
ideal elements in that component) falls back to $\frac{1}{2}$; in particular,
identical alternatives tie at $\mathcal{R}_i = \frac{1}{2}$, as prescribed by
the paper. For $\sigma(F_i) = 1$ the correction term is entirely based on
the membership component, for $\sigma(F_i) = 0$ on the non-membership one, and
intermediate values blend the two. When the hesitation degree vanishes
identically, the index collapses to a convex combination of the fuzzy ranking
indices of He et al. (paper, Theorem 4).

Integrals are computed numerically (trapezoidal rule, `n`-point grid for
$\alpha$, `m`-point for $\beta$; the simplex $T$ of $d_{\mu\nu}$ is enforced
by masking). Overall cost: $O(nN^2)$ for $n$ TrIFNs and $N$ discretisation
steps.

## Layout

| File | Content |
| --- | --- |
| `ifn.py` | `IntuitionisticFuzzyNumber` (trapezoidal $\mu$/$\nu$ params, $\alpha$-, $\beta$-, $(\alpha,\beta)$-cuts, `random_ifn()` random valid instance; validates $b_1 \le a_1 \le b_2 \le a_2 \le a_3 \le b_3 \le a_4 \le b_4$) and `IdealIntuitionisticFuzzyNumber` (constant-interval ideal point) |
| `st_function.py` | `STFunction` — three predefined $s(\alpha)$/$t(\beta)$ weight pairs (linear, exponential, trigonometric) |
| `ifn_distance.py` | `DistanceCalculator` — interval distance $d_I$, component distances $d_\mu$/$d_\nu$/$d_{\mu\nu}$, ideal/anti-ideal cut sets (`f_ast`), $\Delta_\mu$/$\Delta_\nu$, $\sigma$, `ranking_index`, `ranking`; `safe_ratio` ($\frac{1}{2}$ on vanishing denominators), `ranks_to_string` for LaTeX output; `DATA_DIR` |
| `test.py` | Paper Examples 1–7, each run for every $(s, t)$ pair of `STFunction.all_functions()` (linear, exponential, trigonometric), `n = m = 100`; writes per-example JSON and the two LaTeX tables; `look_different_ranking` searches random TrIFN triples whose ranking changes with $(s, t)$ |
| `rankings.json` | Literature rankings (Ye, Rezvani, Roseline–Amirtharaj, Bharati, Mohan, Popa) with DOIs, notation $F_1/F_2/F_3$ |
| `data/` | Generated output (recreated from scratch at every run) |

## Usage

From the repository root (package imports are absolute):

```sh
python -m ranking_ifn.test
```

This **deletes and regenerates** `data/`, then writes:

- `data/example_N.json` — one entry per $(s, t)$ pair (3 per example) with,
  for all the IFNs of the example: the parameters $F_i$, the six distances to
  the ideals, $\Delta_\mu$, $\Delta_\nu$, $\sigma$, the three index terms
  $R_1$, $R_2$, $R_3$, $\mathcal{R}$ and the names of the $s$/$t$ functions
  (N = 1–7)
- `data/table_examples.tex` — LaTeX table of $\sigma$, $R_1$, $R_2$, $R_3$,
  $\mathcal{R}$ and the resulting comparison, one row per $(s, t)$ pair,
  with a `\midrule` between examples (paper Table `tab:examples`)
- `data/table_comparison.tex` — comparison with the literature methods:
  columns from `rankings.json` plus a final *Proposed method* column holding a
  nested table (s-function, t-function, ranking) with one line per $(s, t)$
  pair, derived from `data/example_N.json` (paper Table `tab:comparison`)
- `data/different_ranking.json` — output of `look_different_ranking`: draws
  random triples of TrIFNs (`IntuitionisticFuzzyNumber.random_ifn()`, seed
  101) until the three $(s, t)$ pairs yield **different** rankings, then
  stores the triple (`F1`, `F2`, `F3`) and the distinct ranking strings
  (`ranks`); i.e. a counterexample showing the ranking can depend on the
  choice of $(s, t)$
- `intuitionistic.log` — debug log (written in the current working directory)

Programmatic use:

```python
from ranking_ifn.ifn import IntuitionisticFuzzyNumber
from ranking_ifn.ifn_distance import DistanceCalculator
from ranking_ifn.st_function import STFunction

A = IntuitionisticFuzzyNumber((4, 5, 5.2, 6), (3, 4.8, 5.4, 6.5))
B = IntuitionisticFuzzyNumber((4, 5, 5.5, 6), (4, 5, 5.5, 6.1))

ranks, comparison = DistanceCalculator.ranking(
    s_function=STFunction.s_function_1(),
    t_function=STFunction.t_function_1(),
    fuzzy=[A, B],
    n=100,
    m=100,
)
print(ranks)       # ranking index R for each IFN
print(comparison)  # e.g. "F_{1} > F_{2}"
```

`ranking()` also appends the computed distances, $\Delta_\mu$, $\Delta_\nu$,
$\sigma$ and $\mathcal{R}$ to a JSON file (`file_path` argument, default
`data/ranking_index.json`).

## Requirements

- Python ≥ 3.10
- NumPy ≥ 2.0 (`np.trapezoid`)
