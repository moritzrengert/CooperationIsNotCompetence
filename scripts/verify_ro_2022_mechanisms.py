#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import Decision
from src.games.ro_public_good import public_good_payoff
from src.mechanisms import get_mechanism
from src.mechanisms.ccm import ccm_resolve


def decision(**fields) -> Decision:
    return Decision(
        choice=int(fields.get("choice", 0)),
        prediction=None,
        raw_response=json.dumps(fields, sort_keys=True),
        condition=fields.get("condition"),
        contribution_1=fields.get("contribution_1"),
        threshold_1=fields.get("threshold_1"),
        contribution_2=fields.get("contribution_2"),
        threshold_2=fields.get("threshold_2"),
    )


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_close(actual, expected, label: str, tol: float = 1e-9) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def verify_payoff() -> None:
    assert_close(public_good_payoff(0, 0), 10, "all defect payoff")
    assert_close(public_good_payoff(10, 50), 20, "full contribution payoff")
    assert_close(public_good_payoff(10, 10), 4, "lone contributor payoff")
    assert_close(public_good_payoff(0, 10), 14, "non-contributor with one contributor payoff")


def verify_vcm() -> None:
    mechanism = get_mechanism("vcm")
    decisions = [decision(choice=value) for value in [0, 3, 7, 10, 5]]
    outcome = mechanism.resolve(decisions, {"players": 5, "game_name": "ro_public_good"})
    rows = outcome["rows"]
    assert_equal(outcome["total_contribution"], 25, "VCM total contribution")
    assert_equal([row["implemented_contribution"] for row in rows], [0, 3, 7, 10, 5], "VCM implemented contributions")
    assert_close(rows[0]["payoff"], 20, "VCM player 1 payoff")
    assert_close(rows[3]["payoff"], 10, "VCM player 4 payoff")


def verify_sccm() -> None:
    mechanism = get_mechanism("sccm")
    decisions = [decision(contribution_1=5, threshold_1=25) for _ in range(5)]
    outcome = mechanism.resolve(decisions, {"players": 5, "game_name": "ro_public_good"})
    assert_equal(outcome["total_contribution"], 25, "SCCM coordinated total")
    assert_equal([row["implemented_contribution"] for row in outcome["rows"]], [5, 5, 5, 5, 5], "SCCM implemented contributions")

    infeasible = [
        decision(contribution_1=10, threshold_1=50),
        decision(contribution_1=10, threshold_1=50),
        decision(contribution_1=10, threshold_1=50),
        decision(contribution_1=10, threshold_1=50),
        decision(contribution_1=0, threshold_1=0),
    ]
    outcome = mechanism.resolve(infeasible, {"players": 5, "game_name": "ro_public_good"})
    assert_equal(outcome["total_contribution"], 0, "SCCM infeasible full threshold")


def verify_ccm() -> None:
    messages = [[(5, 25), (10, 50)] for _ in range(5)]
    contributions, total = ccm_resolve(messages)
    assert_equal(total, 50, "CCM highest feasible total")
    assert_equal(contributions, [10, 10, 10, 10, 10], "CCM high offer contributions")

    mechanism = get_mechanism("ccm")
    decisions = [
        decision(contribution_1=5, threshold_1=25, contribution_2=10, threshold_2=50)
        for _ in range(5)
    ]
    outcome = mechanism.resolve(decisions, {"players": 5, "game_name": "ro_public_good"})
    assert_equal(outcome["total_contribution"], 50, "CCM mechanism total")
    assert_close(outcome["rows"][0]["payoff"], 20, "CCM full contribution payoff")


def main() -> int:
    verify_payoff()
    verify_vcm()
    verify_sccm()
    verify_ccm()
    print("PASS RO 2022 mechanisms: payoff, VCM, SCCM, and CCM resolution match the 2022 game.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
