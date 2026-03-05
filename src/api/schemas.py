from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    age:           float = Field(..., ge=18,  le=100,  description="Customer age in years")
    tenure_months: float = Field(..., ge=0,            description="Months as a customer")
    monthly_spend: float = Field(..., ge=0,            description="Avg monthly spend in USD")
    usage_score:   float = Field(..., ge=0,   le=100,  description="Product usage score 0–100")
    clv:           float = Field(..., ge=0,            description="Customer lifetime value")
    am_quality:    float = Field(0.0,                  description="Account manager quality score")

    model_config = {"json_schema_extra": {"example": {
        "age": 35,
        "tenure_months": 24,
        "monthly_spend": 150.0,
        "usage_score": 28.5,
        "clv": 3600.0,
        "am_quality": 0.0,
    }}}


class DecisionResponse(BaseModel):
    recommended_treatment: int
    treatment_label:       str
    cate_estimates:        Dict[str, float]
    expected_lift:         float
    cost_adjusted_lift:    float
    confidence:            float
    rationale:             str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BanditUpdateRequest(BaseModel):
    arm:    int = Field(..., ge=0, le=2, description="Arm index that was pulled")
    reward: int = Field(..., ge=0, le=1, description="1=retained, 0=churned")


class HealthResponse(BaseModel):
    status:        str
    models_loaded: List[str]
    version:       str = "1.0.0"
