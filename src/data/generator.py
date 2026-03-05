import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    n_samples:    int   = 50_000
    random_state: int   = 42
    arm_costs:    list  = field(default_factory=lambda: [0.0, 5.0, 25.0])
    output_dir:   str   = "data/synthetic"
    filename:     str   = "customers.csv"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.filename


class CustomerDataGenerator:
    """
    Generates a synthetic customer retention dataset
    with a known causal structure for treatment effect validation.

    Causal Graph:
        X (covariates) ──────────────────────► T (treatment)
        W (am_quality: unobserved confounder) ► T
        W ───────────────────────────────────► Y (outcome)
        X ───────────────────────────────────► Y
        T ───────────────────────────────────► Y  ← CATE target

    Treatments:
        0 = No Action       (cost: $0)
        1 = Email Campaign  (cost: $5)
        2 = Discount Offer  (cost: $25)
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        np.random.seed(self.config.random_state)
        self._df: Optional[pd.DataFrame] = None

    # ── Private Builders ──────────────────────────────────────────────────────

    def _build_covariates(self, n: int) -> dict:
        age           = np.random.normal(40, 12, n).clip(18, 75)
        tenure_months = np.random.exponential(24, n).clip(1, 120)
        monthly_spend = np.random.lognormal(4.5, 0.8, n).clip(20, 2000)
        usage_score   = np.random.beta(2, 3, n) * 100
        clv           = monthly_spend * tenure_months * np.random.uniform(0.8, 1.2, n)
        am_quality    = np.random.normal(0, 1, n)   # unobserved confounder
        return dict(
            age=age, tenure_months=tenure_months,
            monthly_spend=monthly_spend, usage_score=usage_score,
            clv=clv, am_quality=am_quality
        )

    def _assign_treatment(self, covariates: dict) -> np.ndarray:
        """Biased treatment assignment — higher-value customers get more intervention."""
        clv, usage_score, am_quality = (
            covariates["clv"], covariates["usage_score"], covariates["am_quality"]
        )
        logits = (
            0.3 * (clv - clv.mean()) / clv.std() +
            0.5 * am_quality +
            0.2 * (usage_score - 50) / 50 +
            np.random.gumbel(0, 1, len(clv))
        )
        probs = np.column_stack([
            np.exp(-0.5 * logits),
            np.exp( 0.3 * logits),
            np.exp( 0.8 * logits),
        ])
        probs /= probs.sum(axis=1, keepdims=True)
        return np.array([np.random.choice(3, p=p) for p in probs])

    def _compute_outcome(self, covariates: dict, treatment: np.ndarray) -> tuple:
        """Compute retention outcome and ground-truth causal effects."""
        tenure_months = covariates["tenure_months"]
        usage_score   = covariates["usage_score"]
        am_quality    = covariates["am_quality"]
        clv           = covariates["clv"]

        base_logit = (
            -0.5 +
            0.02 * (tenure_months - 24) / 24 +
            0.01 * (usage_score - 50) +
            0.30 * am_quality +
            0.001 * (clv - 1000) / 1000
        )
        true_effect = np.where(
            treatment == 0, 0.0,
            np.where(
                treatment == 1,
                0.50 + 0.30 * (usage_score < 30),   # email best for low-usage
                1.20 + 0.50 * (clv > 2000)           # discount best for high-CLV
            )
        )
        noise             = np.random.normal(0, 0.3, len(treatment))
        prob_retained     = 1 / (1 + np.exp(-(base_logit + true_effect + noise)))
        customer_retained = np.random.binomial(1, prob_retained)
        return customer_retained, true_effect

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate the full synthetic dataset."""
        n = self.config.n_samples
        logger.info(f"Generating {n:,} synthetic customer records...")

        covariates        = self._build_covariates(n)
        treatment         = self._assign_treatment(covariates)
        retained, effects = self._compute_outcome(covariates, treatment)

        self._df = pd.DataFrame({
            "customer_id":       [f"CUST_{i:06d}" for i in range(n)],
            "age":               covariates["age"].round(1),
            "tenure_months":     covariates["tenure_months"].round(1),
            "monthly_spend":     covariates["monthly_spend"].round(2),
            "usage_score":       covariates["usage_score"].round(2),
            "clv":               covariates["clv"].round(2),
            "am_quality":        covariates["am_quality"].round(4),
            "treatment":         treatment,
            "customer_retained": retained,
            "true_effect":       effects.round(4),
        })
        logger.info("Dataset generation complete.")
        return self._df

    def save(self) -> Path:
        """Persist dataset to CSV."""
        if self._df is None:
            raise RuntimeError("Call generate() before save().")
        out = self.config.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(out, index=False)
        logger.info(f"Saved {len(self._df):,} records → {out}")
        return out

    def summary(self) -> None:
        """Print key dataset statistics."""
        if self._df is None:
            raise RuntimeError("Call generate() before summary().")
        print("\n── Treatment Distribution ──────────────────────")
        print(self._df["treatment"].value_counts().sort_index().to_string())
        print("\n── Retention Rate by Treatment ─────────────────")
        print(self._df.groupby("treatment")["customer_retained"].mean().round(4).to_string())
        print("\n── Mean True Causal Effect by Treatment ────────")
        print(self._df.groupby("treatment")["true_effect"].mean().round(4).to_string())
        print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    config    = DatasetConfig(n_samples=50_000)
    generator = CustomerDataGenerator(config)
    generator.generate()
    generator.save()
    generator.summary()
