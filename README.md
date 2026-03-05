# ⚙ Causal Decision Engine
# ⚙ Causal Decision Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade Causal Machine Learning system that estimates Heterogeneous Treatment Effects (HTE) and recommends optimal, personalized customer retention interventions using causal inference — not just correlation.**

[🚀 Live Demo](https://causal-decision-engine-fxqpeb78e8uptthnskuxwj.streamlit.app/) · [📦 GitHub](https://github.com/PrasanthKumarS777/causal-decision-engine) · [👤 LinkedIn](https://www.linkedin.com/in/prasanthsahu7)

</div>

---

## 📌 Table of Contents

1. [The Problem We Solved](#-the-problem-we-solved)
2. [What Is This Project?](#-what-is-this-project)
3. [Why Causal ML Over Traditional ML?](#-why-causal-ml-over-traditional-ml)
4. [System Architecture](#-system-architecture)
5. [Folder Structure](#-folder-structure)
6. [Models & Algorithms](#-models--algorithms)
   - [T-Learner](#1-t-learner-meta-learner)
   - [Double ML](#2-double-ml-partially-linear-model)
   - [Causal Forest DML](#3-causal-forest-dml)
   - [Thompson Sampling Bandit](#4-thompson-sampling-multi-armed-bandit)
7. [Mathematical Foundations](#-mathematical-foundations)
8. [Why These Models Were Chosen](#-why-these-models-were-chosen)
9. [Dataset & Feature Engineering](#-dataset--feature-engineering)
10. [Training Pipeline](#-training-pipeline)
11. [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
12. [REST API](#-rest-api)
13. [Streamlit Dashboard](#-streamlit-dashboard)
14. [Docker Deployment](#-docker-deployment)
15. [Results & Benchmarks](#-results--benchmarks)
16. [Installation & Setup](#-installation--setup)
17. [Tech Stack](#-tech-stack)
18. [Skills Demonstrated](#-skills-demonstrated)
19. [Author](#-author)

---

## 🔥 The Problem We Solved

### The Business Problem

Customer churn is one of the most costly problems in any subscription or service-based business. Companies routinely spend millions on retention campaigns — emails, discount offers, account manager outreach — but most of these campaigns are **targeted using traditional predictive models** that only answer:

> *"Who is likely to churn?"*

This is fundamentally the **wrong question**. A customer who will churn regardless of any intervention (a "lost cause") and a customer who will stay regardless of any action ("sure thing") both appear high-risk in a standard churn model — but neither benefits from spending retention budget on them.

The right question is:

> *"For this specific customer, which intervention will cause the largest improvement in their probability of staying?"*

### The Statistical Problem

Traditional ML models suffer from **confounding bias** in observational data. In the real world:
- High-value customers are more likely to receive expensive interventions (discount offers)
- Account managers of higher quality are assigned to better customers
- Treatment assignment is not random — it depends on features correlated with the outcome

This means **observed retention rates are confounded**. A naive model sees that customers who received discounts have higher retention, but this is partly because those customers were already more likely to stay — they were selected for the discount because of their value.

**Causal inference corrects for this.** This project estimates the true causal effect of each intervention, removing selection bias, and delivers personalized recommendations grounded in counterfactual reasoning.

---

## 🧠 What Is This Project?

The **Causal Decision Engine** is an end-to-end causal machine learning system that:

1. **Generates** a realistic synthetic dataset of 50,000 customers with observable features, confounders, and treatment assignments
2. **Builds** a causal graph encoding the data-generating process and identifying valid adjustment sets
3. **Trains three causal estimators** — T-Learner, Double ML, and Causal Forest — each estimating the Conditional Average Treatment Effect (CATE) for every customer across three interventions:
   - **No Action** (₹0 cost)
   - **Email Campaign** (₹400 cost)
   - **Discount Offer** (₹2,000 cost)
4. **Deploys a Thompson Sampling multi-armed bandit** that learns from real-time feedback and selects the optimal arm per customer, balancing exploration and exploitation
5. **Serves predictions** via a FastAPI REST endpoint
6. **Visualizes everything** in an interactive Streamlit dashboard with industrial-grade UI
7. **Tracks all experiments** with MLflow for full reproducibility

---

## 🎯 Why Causal ML Over Traditional ML?

| Aspect | Traditional ML | Causal ML (This Project) |
|---|---|---|
| **Question answered** | Who will churn? | What intervention causes retention? |
| **Handles confounding** | ❌ No | ✅ Yes — by design |
| **Personalised effect** | ❌ Population average | ✅ Individual-level CATE |
| **Counterfactual reasoning** | ❌ No | ✅ Yes |
| **Action-optimised** | ❌ Predicts labels | ✅ Recommends interventions with ROI |
| **Bias from observational data** | ❌ Susceptible | ✅ Corrected via Double ML / IPW |
| **Online learning** | ❌ Static | ✅ Thompson Sampling bandit |

> Traditional churn models tell you *who* will churn. Causal models tell you *what to do about it* — and whether doing it will actually make a difference for that specific person.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CAUSAL DECISION ENGINE                      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Data Layer  │───▶│ Causal Graph │───▶│  CATE Estimators │  │
│  │              │    │  (DoWhy)     │    │                  │  │
│  │ CustomerData │    │              │    │  • T-Learner     │  │
│  │ Generator    │    │ Identification    │  • Double ML     │  │
│  │ (50k samples)│    │ & Adjustment │    │  • Causal Forest │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                   │             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────▼─────────┐  │
│  │  MLflow      │◀───│   Training   │◀───│   Model Store    │  │
│  │  Tracking    │    │   Pipeline   │    │  (pkl files)     │  │
│  │  Server      │    │  (train.py)  │    └──────────────────┘  │
│  └──────────────┘    └──────────────┘                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  FastAPI     │    │  Thompson    │    │  Streamlit       │  │
│  │  REST API    │◀───│  Sampling    │───▶│  Dashboard       │  │
│  │  :8000       │    │  Bandit      │    │  :8501           │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Docker Compose Orchestration               │   │
│  │  causal-api (:8000) + streamlit (:8501) + mlflow (:5000)│   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
causal-decision-engine/
│
├── 📂 src/                          # Core source code
│   ├── 📂 api/                      # FastAPI REST API
│   │   ├── main.py                  # API endpoints & app factory
│   │   └── schemas.py               # Pydantic request/response schemas
│   │
│   ├── 📂 causal/                   # Causal inference engine
│   │   ├── estimators.py            # T-Learner, Double ML, Causal Forest
│   │   └── graph.py                 # Causal DAG construction & identification
│   │
│   ├── 📂 data/                     # Data generation layer
│   │   └── generator.py             # Synthetic customer data generator
│   │
│   └── 📂 policy/                   # Decision policy layer
│       └── bandit.py                # Thompson Sampling multi-armed bandit
│
├── 📂 models/                       # Serialised trained models
│   ├── t_learner.pkl                # T-Learner (3 Random Forest arms)
│   ├── double_ml.pkl                # Double ML estimator
│   └── causal_forest.pkl            # Causal Forest DML estimator
│
├── 📂 data/
│   └── 📂 synthetic/
│       └── customers.csv            # Generated dataset (50k rows)
│
├── 📂 mlflow_tracking/              # MLflow experiment store
│   └── 333329047924806730/          # Experiment ID
│       └── <run_id>/                # Per-run metrics, params, artifacts
│           ├── metrics/             # mean_cate, std_cate, etc.
│           ├── params/              # method, base_model, cv_folds, etc.
│           └── artifacts/           # feature_importances.json
│
├── 📂 notebooks/                    # Jupyter exploration notebooks
├── 📂 tests/                        # Unit & integration tests
│   └── test_estimators.py           # CATE estimator test suite
│
├── 📂 .streamlit/
│   └── config.toml                  # Streamlit server configuration
│
├── streamlit_app.py                 # Main Streamlit dashboard (4 tabs)
├── train.py                         # End-to-end training pipeline
├── Dockerfile                       # FastAPI container
├── Dockerfile.streamlit             # Streamlit UI container
├── docker-compose.yml               # 3-service orchestration
├── requirements.txt                 # Python dependencies
├── packages.txt                     # System packages (python3-dev)
├── pytest.ini                       # Test configuration
└── .env                             # Environment variables (gitignored)
```

---

## 🤖 Models & Algorithms

### 1. T-Learner (Meta-Learner)

The T-Learner is a **meta-learning approach** to CATE estimation. It trains a separate outcome model for each treatment arm and estimates the causal effect as the difference between arm predictions.

**Architecture:**
- **3 separate Random Forest Classifiers** — one for each arm (No Action, Email, Discount)
- Base features: `[age, tenure_months, monthly_spend, usage_score, clv, am_quality]`
- Each model predicts `P(retained | features, treatment=k)`

**How it works:**

```
For each customer x:
  mu_0(x) = P(retained | x, T=0)   <- No Action model
  mu_1(x) = P(retained | x, T=1)   <- Email model
  mu_2(x) = P(retained | x, T=2)   <- Discount model

  CATE_email(x)    = mu_1(x) - mu_0(x)
  CATE_discount(x) = mu_2(x) - mu_0(x)
```

**MLflow params logged:** `method=t_learner`, `base_model=random_forest`, `n_estimators`, `max_depth`

**Results:**
- Mean CATE Email: **0.1437**
- Mean CATE Discount: **0.3164**

---

### 2. Double ML (Partially Linear Model)

Double ML (Chernozhukov et al., 2018) is a **semiparametric causal estimator** that achieves robustness to nuisance model misspecification through cross-fitting. It handles confounding by explicitly modelling both the outcome and the treatment propensity.

**Architecture:**
- **Stage 1 — Outcome residualisation:** Train a nuisance model `m(X)` to predict outcome `Y` from features `X`. Compute residuals: `Y_tilde = Y - m_hat(X)`
- **Stage 2 — Treatment residualisation:** Train a nuisance model `g(X)` to predict treatment `T` from features `X`. Compute residuals: `T_tilde = T - g_hat(X)`
- **Stage 3 — CATE estimation:** Regress `Y_tilde` on `T_tilde` using cross-fitting with K folds

**Formula:**

```
Partially Linear Model:
  Y = theta(X) * T + g(X) + epsilon

Where:
  theta(X) = CATE  (the heterogeneous treatment effect)
  g(X)     = baseline outcome function (nuisance)
  epsilon  = noise

Cross-fitting residuals:
  Y_tilde = Y - E[Y|X]
  T_tilde = T - E[T|X]

Final estimate:
  theta_hat(X) = E[Y_tilde * T_tilde] / E[T_tilde^2]
```

**Why cross-fitting?** It prevents overfitting of the nuisance models from biasing the treatment effect estimate — a critical property called **Neyman orthogonality**.

**MLflow params logged:** `method=double_ml`, `cv_folds`

**Results:**
- Mean CATE: **0.2445**

---

### 3. Causal Forest DML

Causal Forest (Wager & Athey, 2018) combined with Double ML cross-fitting is the most powerful estimator in this system. It is a **non-parametric, data-adaptive** method for estimating heterogeneous treatment effects.

**Architecture:**
- Uses **EconML's `CausalForestDML`** implementation
- Internally grows many causal trees, each splitting on features that maximise treatment effect heterogeneity (not outcome prediction)
- Combines Double ML residualisation with forest-based CATE estimation
- Provides **feature importances** for treatment effect heterogeneity, logged as MLflow artifacts

**Key difference from standard Random Forest:**

A standard Random Forest splits on features that best predict `Y`. A Causal Forest splits on features that best explain *variation in treatment effect* `theta(X)` — a fundamentally different and more relevant objective for personalised decision-making.

**Formula:**

```
For a query point x:
  theta_hat(x) = sum_i [ alpha_i(x) * Y_tilde_i ] / sum_i [ alpha_i(x) * T_tilde_i ]

Where:
  alpha_i(x) = kernel weight from forest (1 if leaf co-membership, 0 otherwise)
  Y_tilde_i  = Double ML outcome residual for sample i
  T_tilde_i  = Double ML treatment residual for sample i
```

**MLflow params logged:** `method=causal_forest_dml`, `cv_folds`, `n_estimators`
**MLflow artifacts logged:** `feature_importances.json`

**Results:**
- Mean CATE: **0.2433**

---

### 4. Thompson Sampling Multi-Armed Bandit

The bandit layer sits **on top** of the causal estimators and handles **online policy optimisation** — it learns from real-world feedback (did the customer retain?) and continuously updates its belief about each arm's reward.

**Why a bandit?** Even perfect CATE estimates from offline data may not reflect real deployment performance. The bandit adaptively learns which interventions work best in practice while balancing the **exploration-exploitation trade-off**.

**Algorithm: Thompson Sampling with Beta-Bernoulli conjugate model**

```
Prior for each arm k:
  theta_k ~ Beta(alpha_k, beta_k)   where alpha_0 = beta_0 = 1 (uniform prior)

At decision time:
  1. Sample theta_k_tilde ~ Beta(alpha_k, beta_k) for each arm k
  2. Incorporate CATE signal:
     score_k = theta_k_tilde + lambda * CATE_k - cost_k / CLV
  3. Select arm = argmax_k (score_k)

After observing outcome r in {0, 1}:
  If r = 1 (retained): alpha_k  <- alpha_k + 1
  If r = 0 (churned):  beta_k   <- beta_k  + 1
```

**Key design:** The CATE from offline causal models seeds the bandit's decisions, and real-world feedback corrects it over time. This gives you the best of both worlds — causal rigour offline + adaptive learning online.

**Arm costs:**

| Arm | Intervention | Cost |
|-----|-------------|------|
| 0 | No Action | ₹0 |
| 1 | Email Campaign | ₹400 |
| 2 | Discount Offer | ₹2,000 |

**ROI Calculation:**

```
ROI = (CATE * CLV - Cost) / (Cost + 1) * 100
```

---

## 📐 Mathematical Foundations

### Potential Outcomes Framework (Rubin Causal Model)

The entire project is grounded in the **Potential Outcomes Framework**:

```
For customer i and treatment t in {0, 1, 2}:
  Y_i(t) = potential outcome under treatment t

Observed outcome:
  Y_i = Y_i(T_i)   where T_i is the treatment actually received

Individual Treatment Effect (ITE):
  tau_i = Y_i(1) - Y_i(0)   <- fundamentally unobservable
                                (fundamental problem of causal inference)

Conditional Average Treatment Effect (CATE):
  tau(x) = E[Y(1) - Y(0) | X = x]   <- what we estimate
```

### Identification Assumptions

For CATE to be identifiable from observational data, three assumptions must hold:

```
1. Unconfoundedness (Ignorability):
   Y(t) independent of T given X, for all t
   (No unmeasured confounders given X)

2. Overlap (Positivity):
   0 < P(T=t | X=x) < 1   for all x, t
   (Every customer could have received any treatment)

3. SUTVA (Stable Unit Treatment Value Assumption):
   No interference between units
   (One customer's treatment does not affect another's outcome)
```

The causal graph in `src/causal/graph.py` encodes these assumptions explicitly using DoWhy's DAG framework, and the backdoor adjustment set is identified automatically.

### Backdoor Criterion

The confounder `am_quality` (account manager quality) satisfies the **backdoor criterion**:

```
Causal DAG:
  am_quality  -->  T  (treatment assignment)
  am_quality  -->  Y  (outcome)
  X           -->  T
  X           -->  Y
  T           -->  Y

Adjustment formula:
  P(Y | do(T=t)) = sum_x [ P(Y | T=t, X=x) * P(X=x) ]
```

By conditioning on `am_quality` and `X`, we block all backdoor paths from `T` to `Y`, yielding an unbiased estimate of the causal effect.

---

## 🤔 Why These Models Were Chosen

**T-Learner — for interpretability and baseline**
The T-Learner is the most intuitive approach and serves as the interpretable baseline. By training separate models per treatment arm, it is easy to explain to stakeholders. Its limitation is that it does not explicitly model treatment propensity, making it vulnerable to confounding — addressed by the other two estimators.

**Double ML — for deconfounding with theoretical guarantees**
Double ML was chosen because it has strong theoretical guarantees: it is root-n consistent and asymptotically normal under very weak conditions on the nuisance models. This makes it appropriate for a production system where validity of the causal estimate is paramount. Its cross-fitting procedure ensures that nuisance model overfitting does not contaminate the treatment effect estimate.

**Causal Forest DML — for heterogeneity and non-parametric flexibility**
The Causal Forest is the most powerful estimator for discovering treatment effect heterogeneity — identifying which customer segments respond differently to each intervention. Unlike Double ML which assumes a partially linear model, the Causal Forest makes no parametric assumptions about how CATE varies with features. It is the gold standard for personalised treatment effect estimation in modern causal ML literature.

**Thompson Sampling — for online adaptation**
Thompson Sampling was chosen over epsilon-greedy or UCB bandits because it has better empirical performance in high-variance reward settings and naturally incorporates prior information (the CATE estimates). Its Bayesian posterior update is computationally trivial and interpretable as Beta distributions, which are directly visualised in the dashboard.

---

## 📊 Dataset & Feature Engineering

The synthetic dataset is generated by `src/data/generator.py` using a **structural causal model** that faithfully simulates the real-world data-generating process, including confounding.

### Dataset Size
- **50,000 customers** for training
- **5,000 customers** for dashboard exploration

### Features

| Feature | Type | Description | Role |
|---------|------|-------------|------|
| `age` | Continuous | Customer age (18–75) | Covariate |
| `tenure_months` | Continuous | Months as customer (1–120) | Covariate |
| `monthly_spend` | Continuous | Monthly spend in ₹ (1,500–1,50,000) | Covariate |
| `usage_score` | Continuous | Product usage score (0–100) | Covariate |
| `clv` | Continuous | Customer Lifetime Value in ₹ | Covariate |
| `am_quality` | Continuous | Account manager quality score | **Confounder** |
| `treatment` | Categorical | Assigned intervention (0/1/2) | Treatment |
| `retained` | Binary | Whether customer was retained | Outcome |

### Confounding Mechanism

`am_quality` is a **confounder** — it simultaneously influences treatment assignment and outcome:

```
am_quality --> treatment   (better AMs assign better interventions)
am_quality --> retained    (better AMs retain customers directly)
```

This is why naive correlation between treatment and retention is biased upward — customers who received expensive interventions also had better account managers, inflating apparent treatment efficacy.

### Observed vs Causal Retention Rates

| Treatment | Observed Rate | True CATE | Confounding Bias |
|-----------|--------------|-----------|-----------------|
| No Action | 33.64% | 0.000 | — |
| Email Campaign | 50.30% | 0.1437 | Biased high |
| Discount Offer | 69.11% | 0.3164 | Biased high |

---

## 🔧 Training Pipeline

The `train.py` script executes a 5-step sequential pipeline:

```
Step 1: Data Generation
  └─ CustomerDataGenerator(n_samples=50,000)
  └─ Saves to data/synthetic/customers.csv

Step 2: Causal Graph Construction
  └─ CausalGraphBuilder using DoWhy
  └─ Builds DAG, identifies backdoor adjustment set
  └─ Validates identification assumptions

Step 3: T-Learner Training
  └─ Fits 3 Random Forest classifiers (one per arm)
  └─ Logs mean_cate_email, mean_cate_discount, std to MLflow
  └─ Saves models/t_learner.pkl

Step 4: Double ML Training
  └─ Cross-fitting with K folds
  └─ Logs mean_cate, std_cate to MLflow
  └─ Saves models/double_ml.pkl

Step 5: Causal Forest DML Training
  └─ EconML CausalForestDML with cross-fitting
  └─ Logs mean_cate, std_cate to MLflow
  └─ Saves feature_importances.json as MLflow artifact
  └─ Saves models/causal_forest.pkl
```

**Auto-training:** If `models/t_learner.pkl` is missing at app startup, `streamlit_app.py` automatically triggers `train.py` via subprocess — ensuring the app is always deployable from a clean state.

---

## 📈 MLflow Experiment Tracking

All training runs are tracked in the local MLflow store at `./mlflow_tracking/`.

**Experiment ID:** `333329047924806730`

### Metrics Tracked

| Model | Metrics |
|-------|--------|
| T-Learner | `mean_cate_email`, `mean_cate_discount`, `std_cate_email`, `std_cate_discount` |
| Double ML | `mean_cate`, `std_cate` |
| Causal Forest | `mean_cate`, `std_cate` |

### Parameters Tracked

| Model | Parameters |
|-------|-----------|
| T-Learner | `method`, `base_model`, `n_estimators`, `max_depth` |
| Double ML | `method`, `cv_folds` |
| Causal Forest | `method`, `cv_folds`, `n_estimators` |

### Artifacts Tracked

| Model | Artifact |
|-------|---------|
| Causal Forest | `feature_importances.json` |

**To launch MLflow UI:**
```bash
mlflow ui --backend-store-uri ./mlflow_tracking
# Visit http://localhost:5000
```

Or via Docker:
```bash
docker compose up mlflow
```

---

## 🌐 REST API

The FastAPI service in `src/api/` exposes the causal engine as a REST endpoint.

**Base URL:** `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | CATE prediction + optimal arm |
| `/docs` | GET | Auto-generated Swagger UI |

**Request (`POST /predict`):**

```json
{
  "age": 35,
  "tenure_months": 24,
  "monthly_spend": 12000,
  "usage_score": 28,
  "clv": 288000,
  "am_quality": 0.0
}
```

**Response:**

```json
{
  "recommended_arm": 2,
  "recommended_action": "DISCOUNT OFFER",
  "cate": {
    "no_action": 0.0,
    "email_campaign": 0.1437,
    "discount_offer": 0.3164
  },
  "estimated_lift": 0.3164,
  "intervention_cost_inr": 2000,
  "estimated_roi_pct": 43.6
}
```

---

## 📊 Streamlit Dashboard

The dashboard has **4 interactive tabs**:

**Tab 1 — Decision Center**
Real-time CATE computation from sidebar inputs (age, tenure, spend, usage, CLV, AM quality). Shows the optimal intervention with lift %, causal lift, cost, estimated ROI, and bandit confidence. Bar chart of CATE per arm.

**Tab 2 — CATE Analysis**
CATE distribution histograms for Email and Discount arms across 2,000 sampled customers. Scatter plot of CATE vs CLV coloured by usage score — reveals which customer segments benefit most from each intervention.

**Tab 3 — Bandit Engine**
Live Beta posterior distributions per arm. Alpha/beta parameters and expected reward per arm. Feedback interface to submit real outcomes and update the bandit in real time.

**Tab 4 — Model Comparison**
Bar chart comparing mean CATE across all estimators. Observed vs causal retention rate table with confounding bias warning.

**Live Demo:** [https://causal-decision-engine-fxqpeb78e8uptthnskuxwj.streamlit.app/](https://causal-decision-engine-fxqpeb78e8uptthnskuxwj.streamlit.app/)

---

## 🐳 Docker Deployment

The project ships with a **3-service Docker Compose stack**:

| Service | Container | Port |
|---------|-----------|------|
| FastAPI REST API | `causal-decision-engine-api` | 8000 |
| Streamlit Dashboard | `causal-decision-engine-ui` | 8501 |
| MLflow Tracking Server | `mlflow-server` | 5000 |

### Quick Start

```bash
# Clone the repository
git clone https://github.com/PrasanthKumarS777/causal-decision-engine.git
cd causal-decision-engine

# Build and launch all services
docker compose build
docker compose up -d

# Check logs
docker compose logs -f
```

### Service URLs

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Swagger | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

### Volume Mounts

| Volume | Purpose |
|--------|---------|
| `./mlflow_tracking` | Persists all experiment history |
| `./models` | Shares trained model pickle files |
| `./data` | Shares synthetic dataset |

---

## 📉 Results & Benchmarks

### CATE Estimates by Estimator

| Estimator | Mean CATE | Interpretation |
|-----------|-----------|---------------|
| T-Learner (Email) | 0.1437 | +14.4% retention lift from email |
| T-Learner (Discount) | 0.3164 | +31.6% retention lift from discount |
| Double ML | 0.2445 | Deconfounded, root-n consistent |
| Causal Forest DML | 0.2433 | Non-parametric, heterogeneity-aware |

### Confounding Correction

```
Naive observed effect of Discount:  69.11% - 33.64% = +35.47 pp
True causal effect (CATE):                             +31.64 pp
Confounding bias removed:                               -3.83 pp
```

The system removes ~11% of the apparent treatment effect as confounding bias — preventing over-investment in interventions for customers who would have retained anyway.

### Sample ROI Calculation

```
Customer: CLV = Rs 2,88,000 | Usage = 28 | Tenure = 24 months
Optimal Action: DISCOUNT OFFER
CATE: +31.64%
Cost: Rs 2,000
Estimated ROI: (0.3164 x 2,88,000 - 2,000) / 2,001 x 100 = +4,460%
```

---

## 🚀 Installation & Setup

### Local Setup

```bash
# 1. Clone
git clone https://github.com/PrasanthKumarS777/causal-decision-engine.git
cd causal-decision-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env: set MLFLOW_TRACKING_URI=./mlflow_tracking

# 5. Train models
python train.py

# 6. Launch Streamlit
streamlit run streamlit_app.py

# 7. Launch API (separate terminal)
uvicorn src.api.main:app --reload --port 8000

# 8. Launch MLflow UI (separate terminal)
mlflow ui --backend-store-uri ./mlflow_tracking
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **Causal ML** | EconML 0.16.0, DoWhy 0.14 |
| **ML Framework** | scikit-learn, LightGBM, XGBoost |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit ≥1.35, Plotly 5.19 |
| **Experiment Tracking** | MLflow |
| **Data** | pandas, NumPy, SciPy |
| **Containerisation** | Docker, Docker Compose |
| **Visualisation** | Plotly Express, Plotly Graph Objects |
| **Statistical** | SciPy (Beta distribution for bandit posteriors) |
| **Graph** | NetworkX, pydot |
| **Testing** | pytest |
| **Config** | python-dotenv |

---

## 💡 Skills Demonstrated

**Causal Inference & Econometrics**
- Structural Causal Models (SCMs) and DAG construction with DoWhy
- Potential Outcomes Framework and CATE estimation
- Backdoor criterion, identification, and covariate adjustment
- Double ML — Chernozhukov et al. (2018) with Neyman orthogonality
- Causal Forest DML — Wager & Athey (2018) with heterogeneity detection
- Handling confounding bias in observational data

**Machine Learning Engineering**
- Meta-learner architectures (T-Learner)
- Cross-fitting for nuisance model debiasing
- Multi-armed bandit algorithms with Thompson Sampling
- Bayesian posterior updating (Beta-Bernoulli conjugate model)
- End-to-end ML pipeline design with auto-training

**MLOps & Production Engineering**
- MLflow experiment tracking (metrics, params, artifacts)
- Docker multi-service containerisation with volume persistence
- FastAPI REST API design with Pydantic schemas and Swagger docs
- Streamlit production dashboard deployment on Streamlit Cloud
- Modular, testable, production-ready code architecture

**Data & Statistical Reasoning**
- Synthetic data generation with known causal structure
- Confounding bias quantification and correction
- ROI-driven intervention optimisation
- Feature importance for treatment effect heterogeneity analysis

---

## 👤 Author

**Prasanth Kumar Sahu**

[![GitHub](https://img.shields.io/badge/GitHub-PrasanthKumarS777-181717?style=for-the-badge&logo=github)](https://github.com/PrasanthKumarS777)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-prasanthsahu7-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/prasanthsahu7)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://causal-decision-engine-fxqpeb78e8uptthnskuxwj.streamlit.app/)

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<div align="center">

*Built with rigour, deployed with care.*
*Turning correlation into causation — one customer at a time.*

</div>