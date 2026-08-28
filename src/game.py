"""Backward-compatible imports for older scripts.

New code should import game-specific primitives from `games.*` and mechanism
primitives from `mechanisms.*` directly.
"""

from src.games.fischer_2004 import (
    Treatment,
    best_reply,
    next_reserve,
    payoff,
    payoff_share,
    production,
)
from src.mechanisms.activation import resolve_activation
from src.mechanisms.bccm import (
    bccm_resolve,
    conditional_contribution_resolve as ro_bccm_resolve,
    public_good_payoff as ro_bccm_payoff,
)

__all__ = [
    "Treatment",
    "production",
    "payoff_share",
    "payoff",
    "next_reserve",
    "best_reply",
    "resolve_activation",
    "bccm_resolve",
    "ro_bccm_resolve",
    "ro_bccm_payoff",
]
