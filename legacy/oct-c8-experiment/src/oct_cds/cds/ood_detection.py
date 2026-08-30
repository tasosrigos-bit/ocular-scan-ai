"""Out-of-distribution / not-an-OCT gate.

Runs BEFORE the rule engine. If ``ood_score >= rules.ood_reject_score`` the
case is rejected without an impression. STUB: currently a max-softmax-probability
proxy; replace with a Mahalanobis or energy score fitted on the OCT-C8 train
features before clinical use.
"""

from __future__ import annotations

import numpy as np


def msp_ood_score(probs: dict[str, float] | np.ndarray) -> float:
    """1 - max softmax probability. Higher = more OOD. Cheap, weak baseline."""
    p = np.asarray(list(probs.values()) if isinstance(probs, dict) else probs, dtype=float)
    return float(1.0 - p.max())


def energy_ood_score(logits: np.ndarray, temperature: float = 1.0) -> float:
    """Negative energy score; higher = more OOD. Needs raw logits."""
    logits = np.asarray(logits, dtype=float)
    lse = temperature * np.log(np.sum(np.exp(logits / temperature)))
    return float(-lse)
