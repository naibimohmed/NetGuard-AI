from typing import Any, Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model: Literal["rf", "xgboost"] = "rf"
    features: dict[str, Any]


class PredictionResponse(BaseModel):
    model: str
    prediction: int
    label: str
    attack_probability: float = Field(ge=0, le=1)


# New schemas for CSV / batch prediction
class BatchPredictionRequest(BaseModel):
    model: Literal["rf", "xgboost"] = "rf"
    records: list[dict[str, Any]]


class BatchPredictionResult(BaseModel):
    row: int
    prediction: int
    label: str
    attack_probability: float = Field(ge=0, le=1)


class BatchPredictionResponse(BaseModel):
    model: str
    total: int
    results: list[BatchPredictionResult]