from fastapi import FastAPI
from pydantic import BaseModel
from src.farm_detection.models.predict import Predictor

model = Predictor(model_path="model/gaussiannb.joblib", 
                      preprocessor_path="model/preprocessor.joblib")

class User(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float


app = FastAPI()

@app.post("/predict")
def predict(data: User):
    input_data = [list(data.model_dump().values())]    
    prediction, label = model.predict(input_data)
    return {"prediction": int(prediction[0]), "label": str(label)}


