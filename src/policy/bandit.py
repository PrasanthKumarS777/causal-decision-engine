import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BanditConfig:
    n_arms:      int        = 3
    arm_labels:  List[str]  = field(default_factory=lambda: [
        "No Action", "Email Campaign", "Discount Offer"
    ])
    arm_costs:   List[float] = field(default_factory=lambda: [0.0, 5.0, 25.0])
    cost_weight: float       = 0.1
    cate_weight: float       = 0.3
    explore_weight: float    = 0.6

    def __post_init__(self):
        assert len(self.arm_labels) == self.n_arms, "arm_labels length must match n_arms"
        assert len(self.arm_costs)  == self.n_arms, "arm_costs length must match n_arms"


class ThompsonSamplingBandit:
    """
    Multi-Armed Bandit with Thompson Sampling for cost-adjusted treatment policy.

    Combines:
        - Bayesian exploration (Thompson Sampling via Beta distribution)
        - CATE estimates from EconML for exploitation signal
        - Cost penalty per arm for business-aware decisions

    Score = explore_weight * Thompson_sample
          + cate_weight   * normalized_CATE
          - cost_weight   * normalized_cost
    """

    def __init__(self, config: Optional[BanditConfig] = None):
        self.config   = config or BanditConfig()
        self.alpha    = np.ones(self.config.n_arms)   # successes + 1
        self.beta_    = np.ones(self.config.n_arms)   # failures  + 1
        self._history: List[Dict] = []
        logger.info(
            f"Bandit initialized | Arms={self.config.n_arms} | "
            f"Costs={self.config.arm_costs}"
        )

    # ── Core Methods ──────────────────────────────────────────────────────────

    def select_arm(self, cate: Optional[np.ndarray] = None) -> int:
        """
        Select the best arm using Thompson Sampling + CATE signal.

        Args:
            cate: array of shape (n_arms,) with CATE per arm. If None,
                  falls back to pure Thompson Sampling.
        Returns:
            arm index (int)
        """
        samples = np.random.beta(self.alpha, self.beta_)

        if cate is not None:
            ptp          = cate.max() - cate.min()
            cate_norm    = (cate - cate.min()) / (ptp + 1e-8)
            cost_arr     = np.array(self.config.arm_costs)
            cost_norm    = cost_arr / (cost_arr.max() + 1e-8)
            scores       = (
                self.config.explore_weight * samples +
                self.config.cate_weight    * cate_norm -
                self.config.cost_weight    * cost_norm
            )
        else:
            scores = samples

        return int(np.argmax(scores))

    def update(self, arm: int, reward: int) -> None:
        """
        Update Beta distribution parameters based on observed reward.

        Args:
            arm:    index of the arm that was pulled
            reward: 1 (success / retained) or 0 (failure / churned)
        """
        if reward not in (0, 1):
            raise ValueError(f"Reward must be 0 or 1, got {reward}")
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta_[arm] += 1

        self._history.append({"arm": arm, "reward": reward})
        logger.debug(
            f"Updated arm={arm} ({self.config.arm_labels[arm]}) | "
            f"reward={reward} | alpha={self.alpha[arm]:.0f} | beta={self.beta_[arm]:.0f}"
        )

    def optimal_policy(self, cate_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cost-adjusted optimal arm per customer.

        Args:
            cate_matrix: shape (n_customers, n_arms)
        Returns:
            array of shape (n_customers,) with optimal arm index per customer
        """
        cost_arr = np.array(self.config.arm_costs)
        cost_adj = cate_matrix - cost_arr * self.config.cost_weight
        return np.argmax(cost_adj, axis=1)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def expected_rewards(self) -> np.ndarray:
        """Expected reward per arm = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta_)

    def state(self) -> Dict:
        """Return full bandit state as a serializable dict."""
        exp = self.expected_rewards()
        return {
            "alpha":             self.alpha.tolist(),
            "beta":              self.beta_.tolist(),
            "arm_labels":        self.config.arm_labels,
            "arm_costs":         self.config.arm_costs,
            "expected_rewards":  exp.tolist(),
            "recommended_arm":   int(np.argmax(exp)),
            "recommended_label": self.config.arm_labels[int(np.argmax(exp))],
            "total_updates":     len(self._history),
        }

    def summary(self) -> None:
        """Print bandit state to console."""
        s = self.state()
        print("\n── Bandit State ────────────────────────────────")
        for i, label in enumerate(s["arm_labels"]):
            print(
                f"  Arm {i} | {label:<20} | "
                f"α={self.alpha[i]:.0f} | β={self.beta_[i]:.0f} | "
                f"E[R]={s['expected_rewards'][i]:.4f} | "
                f"Cost=${self.config.arm_costs[i]:.0f}"
            )
        print(f"\n  ✅ Recommended: {s['recommended_label']}")
        print(f"  Total updates: {s['total_updates']}")
        print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    bandit = ThompsonSamplingBandit()

    # Simulate 100 rounds of feedback
    np.random.seed(42)
    true_rates = [0.33, 0.50, 0.69]
    for _ in range(100):
        arm    = bandit.select_arm()
        reward = int(np.random.rand() < true_rates[arm])
        bandit.update(arm, reward)

    bandit.summary()

    # Test with CATE signal
    cate   = np.array([0.0, 0.08, 0.18])
    arm    = bandit.select_arm(cate=cate)
    print(f"CATE-guided selection → Arm {arm}: {bandit.config.arm_labels[arm]}")
