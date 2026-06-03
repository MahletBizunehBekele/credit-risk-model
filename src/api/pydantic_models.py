from pydantic import BaseModel


class PredictionRequest(BaseModel):

    features: dict

class PredictionResponse(BaseModel):
    prediction: int