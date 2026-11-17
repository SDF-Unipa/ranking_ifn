from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())

import numpy as np

from ranking_ifn.comparison_methods import print_comparison
from ranking_ifn.ifn import IntuitionisticFuzzyNumber
from ranking_ifn.ifn_distance import DATA_DIR, DistanceCalculator, ranks_to_string
from ranking_ifn.st_function import STFunction


def example_1(n: int = 100, m: int = 100):
    print("Example 1")
    A = IntuitionisticFuzzyNumber((4, 5, 5.2, 6), (3, 4.8, 5.4, 6.5))
    B = IntuitionisticFuzzyNumber((4, 5, 5.5, 6), (4, 5, 5.5, 6.1))
    file_path: Path = DATA_DIR / "example_1.json"
    for (s_function_name, s_function), (
        t_function_name,
        t_function,
    ) in STFunction.all_functions():
        ranks, ranks_str = DistanceCalculator.ranking(
            s_function=s_function,
            t_function=t_function,
            fuzzy=[A, B],
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        print(
            f"S-Function: {s_function_name}, T-Function: {t_function_name}, Ranks: {ranks}, Comparison: {ranks_str}"
        )
        print_comparison([A, B], example=1)


def example_2(n: int = 100, m: int = 100):
    A = IntuitionisticFuzzyNumber((0.5, 1, 1.5, 2), (0, 0.6, 1.5, 2.5))
    B = IntuitionisticFuzzyNumber((0.5, 1, 1.3, 1.8), (0.2, 0.8, 1.5, 2.5))
    print("\nExample 2")
    file_path: Path = DATA_DIR / "example_2.json"
    for (s_function_name, s_function), (
        t_function_name,
        t_function,
    ) in STFunction.all_functions():
        ranks, ranks_str = DistanceCalculator.ranking(
            s_function=s_function,
            t_function=t_function,
            fuzzy=[A, B],
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        print(
            f"S-Function: {s_function_name}, T-Function: {t_function_name}, Ranks: {ranks}, Comparison: {ranks_str}"
        )
        print_comparison([A, B], example=2)


def example_3(n: int = 100, m: int = 100):
    A = IntuitionisticFuzzyNumber((0.2, 0.25, 0.35, 0.4), (0.1, 0.25, 0.35, 0.75))
    B = IntuitionisticFuzzyNumber((0.2, 0.3, 0.4, 0.5), (0.1, 0.3, 0.4, 0.7))
    print("\nExample 3")
    file_path: Path = DATA_DIR / "example_3.json"

    for (s_function_name, s_function), (
        t_function_name,
        t_function,
    ) in STFunction.all_functions():
        ranks, ranks_str = DistanceCalculator.ranking(
            s_function=s_function,
            t_function=t_function,
            fuzzy=[A, B],
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        print(
            f"S-Function: {s_function_name}, T-Function: {t_function_name}, Ranks: {ranks}, Comparison: {ranks_str}"
        )
        print_comparison([A, B], example=3)


def example_4(n: int = 100, m: int = 100):
    A = IntuitionisticFuzzyNumber((0.2, 0.3, 0.4, 0.5), (0.1, 0.3, 0.4, 0.6))
    B = IntuitionisticFuzzyNumber((0.1, 0.2, 0.3, 0.4), (0.0, 0.2, 0.3, 0.5))
    print("\nExample 4")
    file_path: Path = DATA_DIR / "example_4.json"
    for (s_function_name, s_function), (
        t_function_name,
        t_function,
    ) in STFunction.all_functions():
        ranks, ranks_str = DistanceCalculator.ranking(
            s_function=s_function,
            t_function=t_function,
            fuzzy=[A, B],
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        print(
            f"S-Function: {s_function_name}, T-Function: {t_function_name}, Ranks: {ranks}, Comparison: {ranks_str}"
        )
        print_comparison([A, B], example=4)


def example_5(n: int = 100, m: int = 100):
    A = IntuitionisticFuzzyNumber((4.02, 4.72, 4.72, 4.83), (4.0, 4.72, 4.72, 4.92))
    B = IntuitionisticFuzzyNumber(
        (4.021, 4.721, 4.721, 4.831), (4.01, 4.721, 4.721, 4.921)
    )
    print("\nExample 5")
    file_path: Path = DATA_DIR / "example_5.json"
    for (s_function_name, s_function), (
        t_function_name,
        t_function,
    ) in STFunction.all_functions():
        ranks, ranks_str = DistanceCalculator.ranking(
            s_function=s_function,
            t_function=t_function,
            fuzzy=[A, B],
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        print(
            f"S-Function: {s_function_name}, T-Function: {t_function_name}, Ranks: {ranks}, Comparison: {ranks_str}"
        )
        print_comparison([A, B], example=5)


def example_6(n: int = 100, m: int = 100):
    A = IntuitionisticFuzzyNumber((5.25, 7, 7.05, 8.15), (4.85, 6.95, 7.5, 9.95))
    B = IntuitionisticFuzzyNumber((6.35, 7.15, 7.2, 7.9), (5, 7, 7.25, 9))
    print("\nExample 6")
    file_path: Path = DATA_DIR / "example_6.json"
    for (s_function_name, s_function), (
        t_function_name,
        t_function,
    ) in STFunction.all_functions():
        ranks, ranks_str = DistanceCalculator.ranking(
            s_function=s_function,
            t_function=t_function,
            fuzzy=[A, B],
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        print(
            f"S-Function: {s_function_name}, T-Function: {t_function_name}, Ranks: {ranks}, Comparison: {ranks_str}"
        )
        print_comparison([A, B], example=6)


def example_7(n: int = 100, m: int = 100):
    A1 = IntuitionisticFuzzyNumber((2.95, 5.1, 5.1, 6.65), (2.25, 5.1, 5.1, 9.65))
    A2 = IntuitionisticFuzzyNumber((3.95, 5.15, 5.15, 6.2), (2.325, 5.15, 5.15, 8.75))
    A3 = IntuitionisticFuzzyNumber((3.85, 5.15, 5.15, 6.4), (2.35, 5.15, 5.15, 9.05))
    file_path: Path = DATA_DIR / "example_7.json"
    print("\nExample 7")
    for (s_function_name, s_function), (
        t_function_name,
        t_function,
    ) in STFunction.all_functions():
        ranks, ranks_str = DistanceCalculator.ranking(
            s_function=s_function,
            t_function=t_function,
            fuzzy=[A1, A2, A3],
            n=n,
            m=m,
            file_path=file_path,
            s_function_name=s_function_name,
            t_function_name=t_function_name,
        )
        print(
            f"S-Function: {s_function_name}, T-Function: {t_function_name}, Ranks: {ranks}, Comparison: {ranks_str}"
        )
        print_comparison([A1, A2, A3], example=7)


def main() -> None:
    example_1()
    example_2()
    example_3()
    example_4()
    example_5()
    example_6()
    example_7()


def latex_table(data_dir: Path) -> None:
    latex_table_path: Path = data_dir / f"table_examples.tex"
    pair_keys = [
        "$s$-function",
        "$t$-function",
        "$\\sigma$",
        "$P_1$",
        "$P_2$",
        "$P_3$",
        "$R$",
    ]
    rows: list[str] = []
    print(f"Generating LaTeX table...")
    examples: list[Path] = sorted(data_dir.glob("example_*.json"))
    for t, file_path in enumerate(examples):
        with open(file_path) as f:
            data = json.load(f)
        ex = file_path.stem.split("_")[1]
        for i, row in enumerate(data):
            cells: list[str] = [
                (
                    f"\\multirow{{{len(data)}}}{{*}}{{\\ref{{example:ex-{ex}}}}}"
                    if i == 0
                    else ""
                )
            ]
            for j, k in enumerate(pair_keys):
                if j < 2:
                    cells.append(f"${row[k]}$")
                else:  # example 5 (t == 4) needs more digits in R
                    prec = 6 if (t == 4 and j == len(pair_keys) - 1) else 4
                    cells.append(f"$({', '.join(f'{v:.{prec}f}' for v in row[k])})$")
            cells.append(
                f"${ranks_to_string(np.argsort(row['$R$'])[::-1], row['$R$'])}$"
            )
            rows.append("\t\t" + " & ".join(cells) + " \\\\\n")
        rows.append("\t\t\\midrule\n")
    rows.pop()  # no midrule before bottomrule
    parts: list[str] = []
    parts.append(
        "\\begin{table}[tbh!]\n"
        f"\t\\caption{{Ranking Index for Intuitionistic Fuzzy Numbers. For each example, we provide the tuple of values of $\\sigma$, and $\mathcal{{R}}_{{i}}^{{\\mu\\nu}} = \\tfrac{{d_{{\\mu\\nu}}(F_{{i}},F_{{\\ast}})}}{{d_{{\\mu\\nu}}(F_{{i}},F^{{\\ast}})+d_{{\\mu\\nu}}(F_{{i}},F_{{\\ast}})}}$, $\mathcal{{R}}_i^{{\mu}} = \\tfrac{{d_{{\\mu}}(F_{{i}},F_{{\\ast}})}}{{d_{{\\mu}}(F_{{i}},F^{{\\ast}})+d_{{\\mu}}(F_{{i}},F_{{\\ast}})}}$, $\mathcal{{R}}_i^{{\\nu}} = \\tfrac{{d_{{\\nu}}(F_{{i}},F_{{\\ast}})}}{{d_{{\\nu}}(F_{{i}},F^{{\\ast}})+d_{{\\nu}}(F_{{i}},F_{{\\ast}})}}$, and $\mathcal{{R}}_i = (\mathcal{{R}}_i^{{\\mu\\nu}} + \sigma(F_{{i}})\, \mathcal{{R}}_i^{{\\mu}} + (1-\sigma(F_{{i}}))\, \mathcal{{R}}_i^{{\\nu}})\, /\, 2$  for all the IFNs.}}\n"
        f"\t\\label{{tab:examples}}\n"
        "\t\\centering\n"
        f"\t\\resizebox{{\\textwidth}}{{!}}{{\n"
        "\t\\begin{tabular}{ccccccccc}\n"
        "\t\t\\toprule\n"
        f"\t\t\\textbf{{Example}} & \\textbf{{s-function}} & \\textbf{{t-function}} & $\\mathbf{{\\sigma}}$ & $\\mathbf{{\\mathcal{{R}}}}_i^{{\\mu\\nu}}$ & $\\mathbf{{\\mathcal{{R}}}}_i^{{\\mu}}$ & $\\mathbf{{\\mathcal{{R}}}}_i^{{\\nu}}$ & \\begin{{tabular}}{{@{{}}c@{{}}}}$\\mathbf{{\\mathcal{{R}}}}_i = \\bigl(\\mathbf{{\\mathcal{{R}}}}_i^{{\\mu\\nu}}$ \\\\ $+ \\sigma(F_i)\\, \\mathbf{{\\mathcal{{R}}}}_i^{{\\mu}}$ \\\\ $+ (1 - \\sigma(F_i))\\, \\mathbf{{\\mathcal{{R}}}}_i^{{\\nu}}\\bigr)\\,/\\,2$\\end{{tabular}} & \\textbf{{Comparison}} \\\\\n"
        "\t\t\\midrule\n"
        f"{''.join(rows)}"
        "\t\t\\bottomrule\n"
        "\t\\end{tabular}}\n"
        "\\end{table}"
    )
    latex_table_path.write_text("".join(parts))


def comparison_table(data_dir: Path, rankings_path: Path | None = None) -> None:
    if rankings_path is None:
        rankings_path = data_dir / "rankings.json"
    with open(rankings_path, encoding="utf-8") as f:
        rankings = json.load(f)
    methods: dict[str, dict] = rankings["methods"]

    def cell(latex_ranks: list[str]) -> str:
        if len(latex_ranks) == 1:
            return latex_ranks[0]
        return (
            "\\begin{tabular}{@{}c@{}}" + " \\\\ ".join(latex_ranks) + "\\end{tabular}"
        )

    rows: list[str] = []
    for example, example_data in sorted(
        rankings["examples"].items(), key=lambda kv: int(kv[0])
    ):
        with open(data_dir / f"example_{example}.json") as f:
            data = json.load(f)
        approx = "\\approx"
        proposed: list[str] = []  # one "s & t & ranks" line per (s, t) pair
        seen: set[tuple[str, str, str]] = set()
        for row in data:
            ranks = np.array(row["$R$"])
            ranks_str = ranks_to_string(np.argsort(ranks)[::-1], ranks)
            key_ = (row["$s$-function"], row["$t$-function"], ranks_str)
            if key_ in seen:
                continue
            seen.add(key_)
            s, t, r = key_
            proposed.append(f"${s}$ & ${t}$ & ${r.replace('≈', approx)}$")
        cells: list[str] = [
            cell(["$" + r.replace("≈", approx) + "$" for r in example_data[key]])
            for key in methods
        ]
        cells.append(
            "\\begin{tabular}{@{}ccc@{}}" + " \\\\ ".join(proposed) + "\\end{tabular}"
        )
        rows.append(f"\t\t{example} & " + " & ".join(cells) + " \\\\\n")
        rows.append("\t\t\\midrule\n")

    header: str = " & ".join(
        f"\\textbf{{{m['name']}}} \\cite{{{k}}}" for k, m in methods.items()
    )
    n_cols: int = len(methods) + 2
    table_path: Path = data_dir / "table_comparison.tex"
    table_path.write_text(
        "\\begin{table}[tbh!]\n"
        "\t\\caption{Comparison of the proposed method with other ranking methods for IFNs.}\n"
        "\t\\label{tab:comparison}\n"
        "\t\\centering\n"
        "\t\\resizebox{\\textwidth}{!}{\n"
        f"\t\\begin{{tabular}}{{{'c' * n_cols}}}\n"
        "\t\t\\toprule\n"
        f"\t\t\\textbf{{Example}} & {header} & \\begin{{tabular}}{{@{{}}ccc@{{}}}}\\multicolumn{{3}}{{c}}{{\\textbf{{Proposed method}}}} \\\\ \\textbf{{$s$-function}} & \\textbf{{$t$-function}} & \\textbf{{Ranking}}\\end{{tabular}} \\\\\n"
        "\t\t\\midrule\n"
        f"{''.join(rows[:-1])}"  # Exclude the last midrule
        "\t\t\\bottomrule\n"
        "\t\\end{tabular}}\n"
        "\\end{table}"
    )
    print(f"Comparison table written to {table_path}")


if __name__ == "__main__":
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)  # start from a clean data directory
    DATA_DIR.mkdir(exist_ok=True)
    main()
    latex_table(DATA_DIR)
    comparison_table(DATA_DIR)
