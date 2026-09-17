# -*- coding: utf-8 -*-
"""Kelly gate and bankroll accounting used by the RL trainer.

The model is initialized from scratch. Kelly is used as a purchase-safety signal,
not as a replacement for the RL policy: the policy decides whether/how much to
bet, while actions with non-positive expected value are rejected by default.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class KellyConfig:
    initial_bankroll: int = 100_000
    min_stake: int = 100
    max_bankroll_fraction: float = 0.03
    kelly_fraction_cap: float = 0.03
    min_edge: float = 0.0
    odds_floor: float = 1.01


def implied_kelly(probability: float, decimal_odds: float, config: KellyConfig) -> Tuple[float, float]:
    """Return (full Kelly fraction, expected value per 1 yen)."""
    p = max(0.0, min(1.0, float(probability)))
    odds = max(config.odds_floor, float(decimal_odds))
    b = odds - 1.0
    q = 1.0 - p
    full_kelly = (p * b - q) / b
    expected_value = p * odds - 1.0
    return full_kelly, expected_value


def allowed_stake(bankroll: float, probability: float, decimal_odds: float, config: KellyConfig) -> int:
    """Calculate a legal stake. Return zero when Kelly says not to purchase."""
    full_kelly, expected_value = implied_kelly(probability, decimal_odds, config)
    if expected_value <= config.min_edge or full_kelly <= 0.0 or bankroll < config.min_stake:
        return 0

    fraction = min(full_kelly, config.kelly_fraction_cap, config.max_bankroll_fraction)
    upper = int(bankroll * config.max_bankroll_fraction) // 100 * 100
    stake = int(bankroll * fraction) // 100 * 100
    stake = min(stake, upper)
    return max(config.min_stake, stake) if stake >= config.min_stake else 0


def settle(bankroll: float, stake: int, won: bool, decimal_odds: float) -> float:
    if stake <= 0:
        return float(bankroll)
    return float(bankroll + (stake * (decimal_odds - 1.0) if won else -stake))


def threshold_bonus(previous: float, current: float, initial: int) -> Tuple[float, Dict[int, bool]]:
    bonuses = {2: 1_000, 3: 2_000, 4: 4_000, 5: 8_000, 6: 16_000, 7: 32_000, 8: 64_000, 9: 128_000, 10: 256_000}
    hit = {}
    total = 0.0
    for multiplier, bonus in bonuses.items():
        crossed = previous < initial * multiplier <= current
        hit[multiplier] = crossed
        if crossed:
            total += bonus
    return total, hit
