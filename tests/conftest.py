"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b): neun erfundene Container, in Brüchen rechenbar."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gfm_scenario as S  # noqa: E402


@pytest.fixture
def hand():
    """Neun Container: vier ohne Zollhalt (d = 3,2 / 4,5 / 5,1 / 8,3 Tage), fünf mit Zollhalt (6,5 / 8,2 / 9,7 / 11,4 / 13,1).
    Standtage ⌈d⌉ = 4 / 5 / 6 / 9 und 7 / 9 / 10 / 12 / 14, Standgeld (d > 7) 0 / 0 / 0 / 1 und 0 / 1 / 1 / 1 / 1."""
    return S.HAND_NEUN


@pytest.fixture
def frac():
    return Fraction


class Zufall:
    """Kleine Zufallsquelle für Testinstanzen: SplitMix64 aus gfm_rng (bitgleich auf jeder Plattform, nie numpy.random)."""

    def __init__(self, seed: int):
        import gfm_rng as R
        self.r = R.SplitMix64(seed)

    def random(self, n):
        import numpy as np
        return np.array([self.r.random() for _ in range(n)])

    def uniform(self, lo, hi, n):
        return lo + (hi - lo) * self.random(n)

    def normal(self, n, scale=1.0):
        import numpy as np
        return scale * np.array([self.r.normal() for _ in range(n)])

    def integers(self, lo, hi, n):
        import numpy as np
        return np.array([lo + self.r.below(hi - lo) for _ in range(n)])

    def gamma_int(self, k, mean, n):
        """Gamma mit ganzzahliger Form k und Mittel `mean` (Summe von k Exponentialverteilten)."""
        import numpy as np
        return np.array([sum(self.r.expovariate(mean / k) for _ in range(k)) for _ in range(n)])
