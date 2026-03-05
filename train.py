import logging
import os

from dotenv import load_dotenv
import mlflow

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlflow_tracking"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    from src.data.generator import CustomerDataGenerator, DatasetConfig
    from src.causal.graph import CausalGraphBuilder, CausalGraphConfig
    from src.causal.estimators import (
        EstimatorConfig,
        TLearnerEstimator,
        DoubleMlEstimator,
        CausalForestEstimator,
    )

    # ── Step 1: Generate Data ─────────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("STEP 1 — Generating dataset")
    logger.info("═" * 55)
    config    = DatasetConfig(n_samples=50_000)
    generator = CustomerDataGenerator(config)
    df        = generator.generate()
    generator.save()
    generator.summary()

    # ── Step 2: Build Causal Graph ────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("STEP 2 — Building causal graph")
    logger.info("═" * 55)
    graph_builder = CausalGraphBuilder(CausalGraphConfig())
    graph_builder.build(df).identify()
    graph_builder.summary()

    # ── Step 3: Train T-Learner ───────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("STEP 3 — Training T-Learner")
    logger.info("═" * 55)
    t_learner             = TLearnerEstimator(EstimatorConfig())
    cate_email, cate_disc = t_learner.fit(df)
    logger.info(f"T-Learner complete | MLflow run_id={t_learner.run_id}")

    # ── Step 4: Train Double ML ───────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("STEP 4 — Training Double ML")
    logger.info("═" * 55)
    double_ml  = DoubleMlEstimator(EstimatorConfig())
    cate_dml   = double_ml.fit(df)
    logger.info(f"Double ML complete | MLflow run_id={double_ml.run_id}")

    # ── Step 5: Train Causal Forest ───────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("STEP 5 — Training Causal Forest DML")
    logger.info("═" * 55)
    causal_forest = CausalForestEstimator(EstimatorConfig())
    cate_cf       = causal_forest.fit(df)
    logger.info(f"Causal Forest complete | MLflow run_id={causal_forest.run_id}")

    # ── Step 6: Final Summary ─────────────────────────────────────────────────
    logger.info("═" * 55)
    logger.info("TRAINING COMPLETE — Summary")
    logger.info("═" * 55)
    logger.info(f"T-Learner    | CATE Email={cate_email.mean():.4f} | CATE Discount={cate_disc.mean():.4f}")
    logger.info(f"Double ML    | CATE={cate_dml.mean():.4f} ± {cate_dml.std():.4f}")
    logger.info(f"Causal Forest| CATE={cate_cf.mean():.4f} ± {cate_cf.std():.4f}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  MLflow UI  → mlflow ui --backend-store-uri ./mlflow_tracking")
    logger.info("  API server → uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
