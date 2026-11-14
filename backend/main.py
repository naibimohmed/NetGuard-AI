
from fastapi import FastAPI, HTTPException

from backend.predictor import predict_attack, predict_batch

from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)



app = FastAPI(
    title="NetGuard AI",
    description="Network Intrusion Detection API",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "Welcome to NetGuard AI",
        "status": "running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "available_models": [
            "rf",
            "xgboost"
        ]
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        result = predict_attack(
            request.features,
            request.model
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs."
        )
@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse
)
def predict_batch_endpoint(request: BatchPredictionRequest):

    try:
        if not request.records:
            raise ValueError("No records provided.")

        results = predict_batch(
            records=request.records,
            model_name=request.model
        )

        return {
            "model": request.model,
            "total": len(results),
            "results": results
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed."
        )