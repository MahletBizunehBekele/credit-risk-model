from fastapi import FastAPI

from src.predict import predict

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI()


@app.get("/")
def root():

    return {
        "message":
        "Credit Risk API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_risk(
    request: PredictionRequest
):

    result = predict(
        request.features
    )

    return PredictionResponse(
        prediction=result
    )