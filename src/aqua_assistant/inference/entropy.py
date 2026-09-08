"""Shannon entropy / information-gain math over a posterior distribution."""
from __future__ import annotations

import math


def entropy(distribution: dict[str, float]) -> float:
    """H = -Sum p * log2(p), in bits."""
    total = 0.0
    for p in distribution.values():
        if p > 0:
            total -= p * math.log2(p)
    return total


def normalized_entropy(distribution: dict[str, float]) -> float:
    """Entropy scaled to [0, 1] by log2(N) candidates, for a
    size-independent uncertainty index. 0.0 for 0 or 1 candidates."""
    n = len(distribution)
    if n <= 1:
        return 0.0
    return entropy(distribution) / math.log2(n)
