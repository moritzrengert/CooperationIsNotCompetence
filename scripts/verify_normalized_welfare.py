#!/usr/bin/env python3
"""Verify RO normalized welfare and payoff-efficiency computations.

This script checks that normalized welfare is contribution-equivalent for the
ORS public-good game, that group payoff efficiency agrees with the group
contribution rate, and that repeated chains are not collapsed across
repetitions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from analysis.results import add_core_diagnostics, load_results
from analysis.new_metrics import compute_payoff_efficiency
from src.games.ro_public_good import RO_ENDOWMENT, RO_MPCR, public_good_payoff
from src.mechanisms.ccm import scaled_public_good_payoff

RO_PLAYERS = 5
PI_MIN_TOTAL = RO_PLAYERS * RO_ENDOWMENT
PI_MAX_TOTAL = RO_PLAYERS * (RO_MPCR * RO_ENDOWMENT * RO_PLAYERS)
PI_MAX_PLAYER = RO_MPCR * RO_ENDOWMENT * RO_PLAYERS


def load_simple_metrics_module():
    path = ROOT / "scripts" / "simple_new_metrics.py"
    spec = importlib.util.spec_from_file_location("simple_new_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


simple_metrics = load_simple_metrics_module()


def assert_close(name: str, observed: float, expected: float, tol: float = 1e-9) -> None:
    if pd.isna(observed) or abs(float(observed) - float(expected)) > tol:
        raise AssertionError(f"{name}: observed {observed!r}, expected {expected!r}")


def expected_group_efficiency(payoffs: list[float]) -> float:
    return max(0.0, min(1.0, (sum(payoffs) - PI_MIN_TOTAL) / (PI_MAX_TOTAL - PI_MIN_TOTAL)))


def make_ro_rows(
    mechanism: str,
    payoffs: list[float],
    contribution_units: list[float],
    *,
    repetition: int = 1,
    chain_id: int = 1,
    generation: int = 1,
) -> pd.DataFrame:
    rows = []
    total_units = sum(contribution_units)
    total_contributors = sum(unit > 0 for unit in contribution_units)
    for player, (payoff, units) in enumerate(zip(payoffs, contribution_units), 1):
        contributes = units > 0
        row = {
            "game": f"ro_{mechanism}",
            "mechanism": mechanism,
            "prompt_style": "paper",
            "prompt_modifier": "none",
            "reason_style": "verify",
            "show_generation": False,
            "model": "verifier",
            "temperature": 0.0,
            "seed": 999,
            "treatment": "RESTART",
            "repetition": repetition,
            "chain_id": chain_id,
            "generation": generation,
            "players_in_generation": RO_PLAYERS,
            "player": player,
            "choice": int(contributes),
            "contributes": contributes,
            "total_contributors": total_contributors,
            "payoff": payoff,
            "reason": "constructed verifier row",
            "raw_response": "{}",
        }
        if mechanism == "bccm":
            row["condition"] = 0 if contributes else 5
        if mechanism in {"sccm", "ccm", "ccf", "vcm"}:
            row.update(
                {
                    "implemented_contribution": units,
                    "contribution_share": units / RO_ENDOWMENT,
                    "total_contribution": total_units,
                }
            )
        if mechanism == "sccm":
            row.update(
                {
                    "contribution_1": units,
                    "threshold_1": 0,
                }
            )
        if mechanism == "ccm":
            row.update(
                {
                    "contribution_1": units,
                    "threshold_1": 0,
                    "contribution_2": 0,
                    "threshold_2": 0,
                }
            )
        if mechanism == "ccf":
            row.update(
                {
                    "ccf_source": f"def ccf(others_total):\\n    return {int(units)}\\n",
                    "ccf_min_contribution": units,
                    "ccf_max_contribution": units,
                    "ccf_is_conditional": False,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def check_dataframe(
    label: str,
    df: pd.DataFrame,
    expected_group: float,
    expected_row: float | None = None,
) -> None:
    if expected_row is None:
        expected_row = expected_group
    diag = add_core_diagnostics(df)
    row_level = float(diag["normalized_welfare"].mean())
    simple_eff = simple_metrics.calc_payoff_efficiency(diag)
    gp_df, setup_eff = compute_payoff_efficiency(diag)
    module_eff = float(gp_df["payoff_efficiency"].mean())

    assert_close(f"{label} row-level normalized_welfare", row_level, expected_row)
    assert_close(f"{label} simple_new_metrics.calc_payoff_efficiency", simple_eff, expected_group)
    assert_close(f"{label} analysis.new_metrics.compute_payoff_efficiency", module_eff, expected_group)
    if setup_eff:
        assert_close(f"{label} setup-level efficiency", next(iter(setup_eff.values())), expected_group)

    print(f"PASS {label}: group_E={expected_group:.3f}, row_W={expected_row:.3f}")


def check_constructed_cases() -> None:
    assert public_good_payoff(0, 0) == 10
    assert public_good_payoff(10, 50) == 20
    assert scaled_public_good_payoff(0, 0) == 10
    assert scaled_public_good_payoff(10, 50) == 20

    cases = [
        ("all_defect", [10, 10, 10, 10, 10], [0, 0, 0, 0, 0], None),
        ("all_cooperate", [20, 20, 20, 20, 20], [10, 10, 10, 10, 10], None),
        ("two_contributors", [8, 8, 18, 18, 18], [10, 10, 0, 0, 0], None),
    ]
    for mechanism in ["vcm", "sccm", "ccm", "ccf"]:
        for name, payoffs, units, expected_row in cases:
            expected = expected_group_efficiency(payoffs)
            check_dataframe(f"{mechanism} {name}", make_ro_rows(mechanism, payoffs, units), expected, expected_row)


def check_asymmetric_profile_is_contribution_equivalent() -> None:
    # With one VCM contributor, the contributor earns 4 < defect payoff 10.
    # Normalized welfare must still equal the group contribution rate rather
    # than a clipped individual-payoff average.
    df = make_ro_rows("vcm", [4, 14, 14, 14, 14], [10, 0, 0, 0, 0])
    group_expected = expected_group_efficiency([4, 14, 14, 14, 14])
    check_dataframe("asymmetric contribution equivalence", df, group_expected)


def check_repetition_key_regression() -> None:
    rep1 = make_ro_rows("vcm", [20, 20, 20, 20, 20], [10, 10, 10, 10, 10], repetition=1)
    rep2 = make_ro_rows("vcm", [10, 10, 10, 10, 10], [0, 0, 0, 0, 0], repetition=2)
    combined = pd.concat([rep1, rep2], ignore_index=True)
    check_dataframe("repetition key regression", combined, 0.5)


def check_selected_real_file() -> None:
    path = ROOT / "outputs" / "sweeps" / "ors_2022_open_models_5rep"
    if not path.exists():
        print(f"SKIP selected real-file check: {path} not found; 2022 sweep not complete yet")
        return
    print(f"SKIP selected real-file check: {path} exists but verification is handled by multi-seed selectors.")


def main() -> int:
    check_constructed_cases()
    check_asymmetric_profile_is_contribution_equivalent()
    check_repetition_key_regression()
    check_selected_real_file()
    print("All normalized welfare checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
