import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BanditUpdateRequest,
    BatchRequest,
    CustomerFeatures,
    DecisionResponse,
    HealthResponse,
)
from src.policy.bandit import BanditConfig, ThompsonSamplingBandit

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = ["age", "tenure_months", "monthly_spend", "usage_score", "clv"]
LABELS       = {0: "No Action", 1: "Email Campaign", 2: "Discount Offer"}
store        = {}


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Causal Decision Engine...")
    store["bandit"] = ThompsonSamplingBandit(
        BanditConfig(n_arms=3, arm_costs=[0.0, 5.0, 25.0])
    )
    for name in ["t_learner", "double_ml", "causal_forest"]:
        path = Path(f"models/{name}.pkl")
        if path.exists():
            with open(path, "rb") as f:
                store[name] = pickle.load(f)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model not found: {name} — run train.py first")
    logger.info("✅ API ready.")
    yield
    store.clear()
    logger.info("API shut down cleanly.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Causal Decision Engine",
    description=(
        "Estimates Heterogeneous Treatment Effects (CATE) per customer "
        "and recommends the optimal intervention using causal ML + bandit policy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _features_to_arrays(c: CustomerFeatures):
    X  = np.array([[c.age, c.tenure_months, c.monthly_spend, c.usage_score, c.clv]])
    W  = np.array([[c.am_quality]])
    XW = np.hstack([X, W])
    return X, W, XW


def _compute_cate(X: np.ndarray, W: np.ndarray, XW: np.ndarray) -> dict:
    """Compute CATE using loaded models. Falls back to heuristic if untrained."""
    cate = {0: 0.0}
    if "t_learner" in store:
        arms = store["t_learner"]
        base    = arms[0].predict_proba(XW)[0, 1]
        cate[1] = float(arms[1].predict_proba(XW)[0, 1] - base)
        cate[2] = float(arms[2].predict_proba(XW)[0, 1] - base)
    else:
        # Heuristic fallback before training
        usage  = float(X[0, 3])
        clv    = float(X[0, 4])
        cate[1] = round(0.08 - 0.05 * (usage - 50) / 100, 4)
        cate[2] = round(0.18 + 0.10 * min(clv / 10_000, 1.0), 4)
    return cate


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="healthy",
        models_loaded=[k for k in store if k != "bandit"],
    )


@app.post("/decide", response_model=DecisionResponse, tags=["Decision"])
def decide(c: CustomerFeatures):
    """
    Given a customer's features, return the optimal treatment recommendation
    with CATE estimates, expected lift, and business rationale.
    """
    X, W, XW   = _features_to_arrays(c)
    cate        = _compute_cate(X, W, XW)
    cate_arr    = np.array([cate.get(i, 0.0) for i in range(3)])

    bandit      = store["bandit"]
    arm         = bandit.select_arm(cate=cate_arr)
    exp         = bandit.expected_rewards()
    confidence  = float(exp[arm])
    lift        = float(cate_arr[arm])
    cost_adj    = float(cate_arr[arm] - bandit.config.arm_costs[arm] * 0.001)

    logger.info(f"Decision → arm={arm} ({LABELS[arm]}) | lift={lift:.4f} | conf={confidence:.4f}")

    return DecisionResponse(
        recommended_treatment = arm,
        treatment_label       = LABELS[arm],
        cate_estimates        = {LABELS[k]: round(v, 4) for k, v in cate.items()},
        expected_lift         = round(lift, 4),
        cost_adjusted_lift    = round(cost_adj, 4),
        confidence            = round(confidence, 4),
        rationale             = (
            f"'{LABELS[arm]}' yields an estimated +{lift:.1%} retention lift. "
            f"Customer profile: CLV=${c.clv:.0f}, "
            f"Usage={c.usage_score:.1f}/100, "
            f"Tenure={c.tenure_months:.0f} months."
        ),
    )


@app.post("/decide/batch", response_model=List[DecisionResponse], tags=["Decision"])
def decide_batch(req: BatchRequest):
    """Batch endpoint — accepts up to 1000 customers in one request."""
    return [decide(c) for c in req.customers]


@app.get("/bandit/state", tags=["Bandit"])
def bandit_state():
    """Returns the current Thompson Sampling bandit exploration state."""
    return store["bandit"].state()


@app.post("/bandit/update", tags=["Bandit"])
def bandit_update(req: BanditUpdateRequest):
    """
    Feed observed reward back into the bandit.
    Call this after you observe whether a customer was retained.
    """
    store["bandit"].update(arm=req.arm, reward=req.reward)
    return {
        "updated": True,
        "arm":     req.arm,
        "label":   LABELS[req.arm],
        "state":   store["bandit"].state(),
    }
