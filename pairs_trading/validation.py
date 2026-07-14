"""Validation harness: score the pair selector against a known answer key.

The synthetic generator used to be this repo's central liability -- it silently
backfilled real market data and the published performance numbers came from it. Its
one legitimate use is this: because we know exactly which pairs it built to be
cointegrated, we can ask whether the selector finds them, and whether it also picks
up pairs that are random walks by construction.

That is a test of the selector, not a backtest. Nothing here produces a performance
number, and nothing here may be quoted as one.

Scoring is averaged over many independently seeded universes. A single universe is
one draw: it tells you what happened once, not what the selector does. The first
version of this harness scored one draw and I nearly wrote its false-positive count
into the README as if it were a rate.
"""

from dataclasses import dataclass

import numpy as np

from pairs_trading.cointegration import apply_selection, test_cointegration
from pairs_trading.data import SyntheticDataset, synthetic_dataset

# (correction, criterion, label)
VARIANTS = [
    ("none", "original", "ORIGINAL rule (EG and naive ADF)"),
    ("none", "naive_adf", "naive ADF alone (wrong crit values)"),
    ("none", "mackinnon", "Engle-Granger alone, uncorrected"),
    ("bh", "mackinnon", "Benjamini-Hochberg + EG (default)"),
    ("bonferroni", "mackinnon", "Bonferroni + EG"),
]


@dataclass
class SelectorScore:
    """Confusion-matrix summary of one selection run against the answer key."""

    correction: str
    criterion: str
    selected: list[tuple[str, str]]
    true_positives: list[tuple[str, str]]
    false_positives: list[tuple[str, str]]
    false_negatives: list[tuple[str, str]]
    n_true_pairs: int
    n_null_pairs: int

    @property
    def recall(self) -> float:
        """Share of genuinely cointegrated pairs the selector found."""
        if self.n_true_pairs == 0:
            return 0.0
        return len(self.true_positives) / self.n_true_pairs

    @property
    def false_positive_rate(self) -> float:
        """Share of the non-cointegrated pairs the selector wrongly picked."""
        if self.n_null_pairs == 0:
            return 0.0
        return len(self.false_positives) / self.n_null_pairs

    @property
    def precision(self) -> float:
        """Share of selected pairs that are genuinely cointegrated."""
        if not self.selected:
            return 0.0
        return len(self.true_positives) / len(self.selected)


def score_selector(
    dataset: SyntheticDataset,
    significance: float = 0.05,
    correction: str = "bh",
    criterion: str = "mackinnon",
    results: list[dict] | None = None,
) -> SelectorScore:
    """Run one selection rule on labelled synthetic data and score it against truth.

    max_pairs is set to every possible pair, deliberately: capping at 3 would hide
    false positives behind the cap instead of measuring them.
    """
    all_pairs = dataset.all_pairs
    truth = {frozenset(p) for p in dataset.cointegrated_pairs}

    if results is None:
        results = test_cointegration(dataset.prices, significance)

    selected = apply_selection(
        results,
        significance=significance,
        max_pairs=len(all_pairs),
        correction=correction,
        criterion=criterion,
    )
    selected_sets = {frozenset(p) for p in selected}

    return SelectorScore(
        correction=correction,
        criterion=criterion,
        selected=selected,
        true_positives=[p for p in selected if frozenset(p) in truth],
        false_positives=[p for p in selected if frozenset(p) not in truth],
        false_negatives=[
            p for p in dataset.cointegrated_pairs if frozenset(p) not in selected_sets
        ],
        n_true_pairs=len(dataset.cointegrated_pairs),
        n_null_pairs=len(dataset.null_pairs),
    )


def run_validation(
    tickers: list[str],
    start: str,
    end: str,
    significance: float = 0.05,
    n_universes: int = 20,
    strict: bool = True,
) -> dict:
    """Score the selector on `n_universes` independently seeded synthetic universes.

    Asserts, when `strict`, that the default selector (Benjamini-Hochberg +
    Engle-Granger):
      1. recovers the pairs that are cointegrated by construction, and
      2. selects strictly fewer false positives than the rule this repo shipped.

    Raises AssertionError otherwise. Returns the aggregated scores.
    """
    reference = synthetic_dataset(tickers, start, end, seed=0)

    print("=" * 78)
    print("SELECTOR VALIDATION ON LABELLED SYNTHETIC DATA")
    print("=" * 78)
    print("\nSimulated data with a KNOWN answer key. This is a unit test of the pair")
    print("selector. It is NOT a backtest and yields no performance claim.\n")
    print(f"Universes:      {n_universes} independent seeds")
    print(f"Each universe:  {len(reference.prices.columns)} tickers, "
          f"{len(reference.prices)} days, {len(reference.all_pairs)} pairs tested")
    print(f"Cointegrated by construction ({len(reference.cointegrated_pairs)}): "
          f"{[f'{a}/{b}' for a, b in reference.cointegrated_pairs]}")
    print(f"Random-walk spread by construction ({len(reference.independent_pairs)}): "
          f"{[f'{a}/{b}' for a, b in reference.independent_pairs]}")
    print(f"Null pairs (answer key for false positives): "
          f"{len(reference.null_pairs)} per universe")

    agg: dict[tuple[str, str], dict[str, list]] = {
        (c, k): {"tp": [], "fp": [], "sel": []} for c, k, _ in VARIANTS
    }

    for seed in range(n_universes):
        dataset = synthetic_dataset(tickers, start, end, seed=seed)
        # One set of tests per universe; every rule is scored off the same p-values.
        results = test_cointegration(dataset.prices, significance)
        for correction, criterion, _label in VARIANTS:
            score = score_selector(
                dataset,
                significance=significance,
                correction=correction,
                criterion=criterion,
                results=results,
            )
            agg[(correction, criterion)]["tp"].append(len(score.true_positives))
            agg[(correction, criterion)]["fp"].append(len(score.false_positives))
            agg[(correction, criterion)]["sel"].append(len(score.selected))

    n_true = len(reference.cointegrated_pairs)
    n_null = len(reference.null_pairs)

    print("\n" + "-" * 78)
    print(f"Mean over {n_universes} universes")
    print(f"{'selection rule':<42}{'picked':>7}{'true':>7}{'false':>7}"
          f"{'recall':>8}{'FP rate':>8}")
    print("-" * 78)

    summary = {}
    for correction, criterion, label in VARIANTS:
        a = agg[(correction, criterion)]
        tp, fp, sel = np.mean(a["tp"]), np.mean(a["fp"]), np.mean(a["sel"])
        summary[(correction, criterion)] = {"tp": tp, "fp": fp, "sel": sel}
        print(f"{label:<42}{sel:>7.2f}{tp:>7.2f}{fp:>7.2f}"
              f"{tp / n_true:>7.0%}{fp / n_null:>7.1%}")

    print("-" * 78)
    print(f"'recall'  = share of the {n_true} truly cointegrated pairs found.")
    print(f"'FP rate' = share of the {n_null} null pairs wrongly selected.")

    original = summary[("none", "original")]
    default = summary[("bh", "mackinnon")]

    print("\nNote on the residual false-positive rate under BH: it does not fall to")
    print("zero, and it should not be expected to. The 45 pairwise tests are drawn from")
    print("only 10 series, so they are strongly dependent -- if one series happens to")
    print("wander in a mean-reverting way, every pair containing it is pulled toward")
    print("rejection together. Benjamini-Hochberg controls FDR under independence or")
    print("positive regression dependence, and that assumption is not satisfied here.")
    print("BH still cuts the false-positive count substantially; it does not abolish it.")

    print("\n" + "=" * 78)
    if strict:
        assert default["tp"] > 0, (
            "Default selector (BH + Engle-Granger) recovered none of the "
            f"{n_true} pairs that are cointegrated by construction."
        )
        assert default["fp"] < original["fp"], (
            "Default selector did not reduce false positives relative to the original "
            f"rule: {default['fp']:.2f} vs {original['fp']:.2f} per universe."
        )
        print(f"PASS: default selector recovers {default['tp']:.2f}/{n_true} true pairs "
              f"per universe")
        print(f"      and cuts mean false positives from {original['fp']:.2f} "
              f"(original rule) to {default['fp']:.2f}.")
    print("=" * 78)

    return summary
