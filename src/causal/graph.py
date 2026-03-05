import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dowhy import CausalModel

logger = logging.getLogger(__name__)


@dataclass
class CausalGraphConfig:
    treatment:  str = "treatment"
    outcome:    str = "customer_retained"
    common_causes: list = None

    def __post_init__(self):
        if self.common_causes is None:
            self.common_causes = ["am_quality", "clv", "usage_score"]


CAUSAL_GRAPH = """
digraph {
    age -> treatment;
    age -> customer_retained;
    tenure_months -> treatment;
    tenure_months -> customer_retained;
    monthly_spend -> treatment;
    monthly_spend -> customer_retained;
    usage_score -> treatment;
    usage_score -> customer_retained;
    clv -> treatment;
    clv -> customer_retained;
    am_quality -> treatment;
    am_quality -> customer_retained;
    treatment -> customer_retained;
}
"""


class CausalGraphBuilder:
    """
    Builds and identifies the causal model using DoWhy.

    Responsibilities:
        - Define the causal DAG (Directed Acyclic Graph)
        - Identify the estimand (what can be estimated from data)
        - Expose the model for downstream CATE estimators
    """

    def __init__(self, config: Optional[CausalGraphConfig] = None):
        self.config = config or CausalGraphConfig()
        self._model:    Optional[CausalModel] = None
        self._estimand: Optional[object]      = None

    def build(self, df: pd.DataFrame) -> "CausalGraphBuilder":
        """Construct the DoWhy CausalModel from the dataset."""
        logger.info("Building causal graph...")
        self._model = CausalModel(
            data=df,
            treatment=self.config.treatment,
            outcome=self.config.outcome,
            graph=CAUSAL_GRAPH.strip(),
            common_causes=self.config.common_causes,
        )
        logger.info("Causal graph built successfully.")
        return self

    def identify(self, proceed_when_unidentifiable: bool = True) -> "CausalGraphBuilder":
        """Identify the causal estimand from the graph."""
        if self._model is None:
            raise RuntimeError("Call build() before identify().")
        logger.info("Identifying causal estimand...")
        self._estimand = self._model.identify_effect(
            proceed_when_unidentifiable=proceed_when_unidentifiable
        )
        logger.info("Estimand identified.")
        return self

    @property
    def model(self) -> CausalModel:
        if self._model is None:
            raise RuntimeError("Model not built yet. Call build() first.")
        return self._model

    @property
    def estimand(self):
        if self._estimand is None:
            raise RuntimeError("Estimand not identified yet. Call identify() first.")
        return self._estimand

    def summary(self) -> None:
        """Print the identified estimand."""
        print("\n── Identified Estimand ─────────────────────────")
        print(self._estimand)
        print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    from src.data.generator import CustomerDataGenerator, DatasetConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df      = CustomerDataGenerator(DatasetConfig(n_samples=5_000)).generate()
    builder = CausalGraphBuilder()
    builder.build(df).identify()
    builder.summary()
