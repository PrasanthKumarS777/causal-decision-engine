import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logger = logging.getLogger(__name__)

FEATURE_COLS    = ["age", "tenure_months", "monthly_spend", "usage_score", "clv"]
CONFOUNDER_COLS = ["am_quality"]


@dataclass
class EstimatorConfig:
    experiment_name: str        = "causal_decision_engine"
    models_dir:      str        = "models"
    n_estimators:    int        = 100
    max_depth:       int        = 4
    cv_folds:        int        = 3
    random_state:    int        = 42
    feature_cols:    list       = field(default_factory=lambda: FEATURE_COLS)
    confounder_cols: list       = field(default_factory=lambda: CONFOUNDER_COLS)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)


class BaseEstimator:
    """Base class for all causal estimators."""

    def __init__(self, config: Optional[EstimatorConfig] = None):
        self.config  = config or EstimatorConfig()
        self._model  = None
        self._run_id: Optional[str] = None
        mlflow.set_tracking_uri("./mlflow_tracking")
        mlflow.set_experiment(self.config.experiment_name)
        self.config.models_path.mkdir(parents=True, exist_ok=True)

    def _extract_matrices(self, df: pd.DataFrame):
        """Extract Y, T, X, W matrices from dataframe."""
        Y  = df["customer_retained"].values.astype(float)
        T  = df["treatment"].values.astype(float)
        X  = df[self.config.feature_cols].values
        W  = df[self.config.confounder_cols].values
        XW = np.hstack([X, W])
        return Y, T, X, W, XW

    def _save_model(self, filename: str) -> Path:
        path = self.config.models_path / filename
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, filepath: str) -> object:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id


class TLearnerEstimator(BaseEstimator):
    """
    T-Learner: trains a separate outcome model per treatment arm.
    CATE = μ_t(x) - μ_0(x)  for each treatment arm t.
    """

    def fit(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Training T-Learner...")
        Y, T, X, W, XW = self._extract_matrices(df)

        with mlflow.start_run(run_name="T-Learner") as run:
            self._run_id = run.info.run_id
            arms = {}
            for t in [0, 1, 2]:
                model = GradientBoostingClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state
                )
                model.fit(XW[T == t], Y[T == t])
                arms[t] = model

            self._model = arms
            base       = arms[0].predict_proba(XW)[:, 1]
            cate_email = arms[1].predict_proba(XW)[:, 1] - base
            cate_disc  = arms[2].predict_proba(XW)[:, 1] - base

            mlflow.log_params({
                "method":       "T-Learner",
                "base_model":   "GradientBoostingClassifier",
                "n_estimators": self.config.n_estimators,
                "max_depth":    self.config.max_depth,
            })
            mlflow.log_metrics({
                "mean_cate_email":    float(cate_email.mean()),
                "std_cate_email":     float(cate_email.std()),
                "mean_cate_discount": float(cate_disc.mean()),
                "std_cate_discount":  float(cate_disc.std()),
            })

        self._save_model("t_learner.pkl")
        logger.info(f"T-Learner | CATE Email={cate_email.mean():.4f} | CATE Discount={cate_disc.mean():.4f}")
        return cate_email, cate_disc

    def predict(self, X: np.ndarray, W: np.ndarray) -> dict:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        XW   = np.hstack([X, W])
        base = self._model[0].predict_proba(XW)[:, 1]
        return {
            "cate_email":    self._model[1].predict_proba(XW)[:, 1] - base,
            "cate_discount": self._model[2].predict_proba(XW)[:, 1] - base,
        }


class DoubleMlEstimator(BaseEstimator):
    """
    Double ML (LinearDML): removes confounding via cross-fitting.
    Robust to regularization bias in high-dimensional settings.
    """

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        logger.info("Training Double ML...")
        Y, T, X, W, _ = self._extract_matrices(df)
        T_bin = (T > 0).astype(float)

        with mlflow.start_run(run_name="Double-ML") as run:
            self._run_id = run.info.run_id
            self._model  = LinearDML(
                model_y=GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state
                ),
                random_state=self.config.random_state,
                cv=self.config.cv_folds,
            )
            self._model.fit(Y, T_bin, X=X, W=W)
            cate = self._model.effect(X)

            mlflow.log_params({
                "method":   "Double-ML",
                "cv_folds": self.config.cv_folds,
            })
            mlflow.log_metrics({
                "mean_cate": float(cate.mean()),
                "std_cate":  float(cate.std()),
            })

        self._save_model("double_ml.pkl")
        logger.info(f"Double ML | Mean CATE={cate.mean():.4f} | Std={cate.std():.4f}")
        return cate

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.effect(X)


class CausalForestEstimator(BaseEstimator):
    """
    Causal Forest DML: non-parametric heterogeneous treatment effect estimator.
    Provides per-customer CATE with confidence intervals.
    """

    def fit(self, df: pd.DataFrame) -> np.ndarray:
        logger.info("Training Causal Forest DML...")
        Y, T, X, W, _ = self._extract_matrices(df)
        T_bin = (T > 0).astype(float)

        with mlflow.start_run(run_name="Causal-Forest-DML") as run:
            self._run_id = run.info.run_id
            self._model  = CausalForestDML(
                n_estimators=200,
                min_samples_leaf=5,
                model_y=GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=self.config.random_state
                ),
                random_state=self.config.random_state,
                cv=self.config.cv_folds,
            )
            self._model.fit(Y, T_bin, X=X, W=W)
            cate        = self._model.effect(X)
            importances = dict(zip(
                self.config.feature_cols,
                self._model.feature_importances_.tolist()
            ))

            mlflow.log_params({
                "method":       "Causal-Forest-DML",
                "n_estimators": 200,
                "cv_folds":     self.config.cv_folds,
            })
            mlflow.log_metrics({
                "mean_cate": float(cate.mean()),
                "std_cate":  float(cate.std()),
            })
            mlflow.log_dict(importances, "feature_importances.json")

        self._save_model("causal_forest.pkl")
        logger.info(f"Causal Forest | Mean CATE={cate.mean():.4f} | Std={cate.std():.4f}")
        return cate

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.effect(X)

    def effect_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower_bound, upper_bound) confidence interval."""
        lb, ub = self._model.effect_interval(X, alpha=0.05)
        return lb, ub
